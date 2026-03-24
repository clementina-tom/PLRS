import streamlit as st
import torch
import torch.nn as nn
import json
import pandas as pd
import networkx as nx
import numpy as np
from huggingface_hub import hf_hub_download
from typing import Dict, List, Optional

st.set_page_config(page_title='Logic Engine', page_icon='🧠', layout='wide')

HF_REPO = 'Clementio/PLRS'

@st.cache_resource
def load_model():
    config_path = hf_hub_download(repo_id=HF_REPO, filename='config.json')
    with open(config_path) as f:
        config = json.load(f)
    model_path = hf_hub_download(repo_id=HF_REPO, filename='sakt_model.pt')
    class SAKT(nn.Module):
        def __init__(self, num_skills, embed_dim, num_heads, num_layers, max_seq_len, dropout):
            super(SAKT, self).__init__()
            self.num_skills = num_skills
            self.interaction_embed = nn.Embedding(num_skills * 2 + 1, embed_dim, padding_idx=0)
            self.skill_embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)
            self.pos_embed = nn.Embedding(max_seq_len + 1, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, dim_feedforward=embed_dim * 4, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(embed_dim, 1)
        def forward(self, interactions, target_skills, mask):
            batch_size, seq_len = interactions.shape
            positions = torch.arange(seq_len, device=interactions.device).unsqueeze(0).expand(batch_size, -1)
            x = self.interaction_embed(interactions)
            x = x + self.pos_embed(positions)
            x = x * mask.unsqueeze(-1).float()
            x = self.dropout(x)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            x = self.transformer(x, mask=causal_mask, is_causal=False)
            x = x * mask.unsqueeze(-1).float()
            x = x + self.skill_embed(target_skills)
            return self.output(x).squeeze(-1)
    device = torch.device('cpu')
    model = SAKT(num_skills=config['num_skills'], embed_dim=config['embed_dim'], num_heads=config['num_heads'], num_layers=config['num_layers'], max_seq_len=config['max_seq_len'], dropout=config['dropout'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config, device

@st.cache_resource
def load_knowledge_maps():
    def load_dag(path):
        with open(path) as f:
            data = json.load(f)
        G = nx.DiGraph()
        for node in data['nodes']:
            G.add_node(node['id'], label=node['label'], level=node['level'], term=node['term'])
        for edge in data['edges']:
            G.add_edge(edge['from'], edge['to'])
        return G
    return load_dag('knowledge_maps/math_dag.json'), load_dag('knowledge_maps/cs_dag.json')

@st.cache_data
def load_skill_encoder():
    return pd.read_csv('data/skill_encoder.csv')

class MasteryVector:
    def __init__(self, graph, threshold=0.70):
        self.graph = graph
        self.threshold = threshold
        self.mastery = {node: 0.0 for node in graph.nodes}
    def update(self, topic_id, probability):
        if topic_id in self.mastery: self.mastery[topic_id] = probability
    def is_mastered(self, topic_id):
        return self.mastery.get(topic_id, 0.0) >= self.threshold
    def get_mastery(self, topic_id):
        return self.mastery.get(topic_id, 0.0)
    def get_mastery_summary(self):
        mastered = [t for t in self.mastery if self.is_mastered(t)]
        return {'total_topics': len(self.mastery), 'mastered': len(mastered), 'mastery_rate': round(len(mastered)/len(self.mastery), 3), 'mastered_topics': mastered}

class DAGConstraintLayer:
    def __init__(self, graph, threshold=0.70):
        self.graph = graph
        self.threshold = threshold
    def validate(self, topic_id, mastery_vector):
        if topic_id not in self.graph.nodes: return False, 'Topic not found.'
        prerequisites = list(self.graph.predecessors(topic_id))
        label = self.graph.nodes[topic_id].get('label', topic_id)
        if not prerequisites: return True, f'✅ Foundational topic — no prerequisites.'
        unmastered = [(self.graph.nodes[p].get('label',p), mastery_vector.get_mastery(p)) for p in prerequisites if not mastery_vector.is_mastered(p)]
        if unmastered:
            gaps = ', '.join([f"{lbl} ({m:.0%} mastered, need {self.threshold:.0%})" for lbl,m in unmastered])
            return False, f'❌ Prerequisites not met: {gaps}'
        prereq_labels = [self.graph.nodes[p].get('label',p) for p in prerequisites]
        return True, f'✅ Prerequisites mastered: {", ".join(prereq_labels)}'

class RankingFunction:
    def __init__(self, graph, threshold=0.70, w_gap=0.40, w_ready=0.35, w_downstream=0.25):
        self.graph=graph; self.threshold=threshold; self.w_gap=w_gap; self.w_ready=w_ready; self.w_downstream=w_downstream
        scores = {n: len(nx.descendants(graph, n)) for n in graph.nodes}
        mx = max(scores.values()) if scores else 1
        self._downstream = {n: s/mx for n,s in scores.items()}
    def score(self, topic_id, mastery_vector):
        current = mastery_vector.get_mastery(topic_id)
        gap = min(max(0.0, self.threshold-current)/self.threshold, 1.0)
        prereqs = list(self.graph.predecessors(topic_id))
        readiness = 1.0 if not prereqs else sum(1 for p in prereqs if mastery_vector.is_mastered(p))/len(prereqs)
        downstream = self._downstream.get(topic_id, 0.0)
        return round(self.w_gap*gap + self.w_ready*readiness + self.w_downstream*downstream, 3)

class LearningRecommendationPipeline:
    def __init__(self, graph, threshold=0.70, top_n=5):
        self.graph=graph; self.constraint=DAGConstraintLayer(graph,threshold); self.ranker=RankingFunction(graph,threshold); self.top_n=top_n
    def run(self, mastery_vector):
        approved, vetoed = [], []
        for topic_id in self.graph.nodes:
            is_approved, reasoning = self.constraint.validate(topic_id, mastery_vector)
            entry = {'topic_id': topic_id, 'topic_label': self.graph.nodes[topic_id].get('label', topic_id), 'mastery': round(mastery_vector.get_mastery(topic_id),3), 'reasoning': reasoning, 'approved': is_approved}
            if is_approved and not mastery_vector.is_mastered(topic_id):
                entry['score'] = self.ranker.score(topic_id, mastery_vector)
                approved.append(entry)
            elif not is_approved: vetoed.append(entry)
        approved.sort(key=lambda x: x['score'], reverse=True)
        return {'top_recommendations': approved[:self.top_n], 'total_approved': len(approved), 'total_vetoed': len(vetoed), 'vetoed_sample': vetoed[:5], 'prerequisite_violation_rate': round(len(vetoed)/max(len(list(self.graph.nodes)),1),3)}

ACTIVITY_TO_MATH = {'oucontent':'algebraic_expressions','forumng':'statistics_basic','homepage':'whole_numbers','subpage':'plane_shapes','resource':'indices','url':'number_bases','ouwiki':'proportion_variation','glossary':'algebraic_factorization','quiz':'quadratic_equations'}
ACTIVITY_TO_CS   = {'oucontent':'programming_concepts','forumng':'ethics_technology','homepage':'computer_basics','subpage':'html_basics','resource':'networking_fundamentals','url':'internet_basics','ouwiki':'cloud_basics','glossary':'intro_databases','quiz':'python_basics'}

def run_sakt_inference(model, config, skill_seq, correct_seq, device):
    max_len=config['max_seq_len']; n_skills=config['num_skills']
    if len(skill_seq)>max_len: skill_seq=skill_seq[-max_len:]; correct_seq=correct_seq[-max_len:]
    interactions=[s+c*n_skills for s,c in zip(skill_seq[:-1],correct_seq[:-1])]
    target_skills=skill_seq[1:]
    seq_len=len(interactions); pad_len=max_len-seq_len
    interactions=[0]*pad_len+interactions; target_skills=[0]*pad_len+target_skills; mask=[False]*pad_len+[True]*seq_len
    with torch.no_grad():
        logits=model(torch.LongTensor([interactions]).to(device),torch.LongTensor([target_skills]).to(device),torch.BoolTensor([mask]).to(device))
        probs=torch.sigmoid(logits).squeeze(0)
    mastery={}; real_probs=probs[torch.BoolTensor(mask)].cpu().numpy(); real_skills=target_skills[pad_len:]
    for skill_id,prob in zip(real_skills,real_probs): mastery[int(skill_id)]=float(prob)
    return mastery

def build_mastery_vector(skill_probs, graph, skill_encoder_df, domain, threshold):
    mv=MasteryVector(graph,threshold); mapping=ACTIVITY_TO_MATH if domain=='math' else ACTIVITY_TO_CS
    topic_scores={}
    for skill_id,prob in skill_probs.items():
        row=skill_encoder_df[skill_encoder_df['skill_id']==skill_id]
        if row.empty: continue
        act=row['activity_type'].values[0] if 'activity_type' in row.columns else None
        topic_id=mapping.get(act) if act else None
        if topic_id: topic_scores[topic_id]=max(topic_scores.get(topic_id,0.0),prob)
    for topic_id,score in topic_scores.items(): mv.update(topic_id,score)
    return mv

def main():
    model, config, device = load_model()
    math_graph, cs_graph  = load_knowledge_maps()
    skill_encoder         = load_skill_encoder()
    st.title('🧠 Logic Engine')
    st.subheader('Domain-Agnostic Constraint-Aware Learning Recommender')
    st.markdown('---')
    st.sidebar.title('⚙️ Configuration')
    domain    = st.sidebar.selectbox('Select Domain', ['Mathematics', 'CS Fundamentals'])
    threshold = st.sidebar.slider('Mastery Threshold', 0.50, 0.90, 0.70, 0.05)
    top_n     = st.sidebar.slider('Top N Recommendations', 3, 10, 5)
    graph      = math_graph if domain=='Mathematics' else cs_graph
    domain_key = 'math'     if domain=='Mathematics' else 'cs'
    pipeline   = LearningRecommendationPipeline(graph, threshold, top_n)
    st.sidebar.markdown('---')
    st.sidebar.markdown('**About**')
    st.sidebar.markdown('SAKT-based knowledge tracing with DAG prerequisite constraints.')
    tab1, tab2, tab3 = st.tabs(['🎯 Get Recommendations','🗺️ Knowledge Map','📊 Diagnostics'])
    with tab1:
        st.header('Learner Profile')
        mode = st.radio('Input Mode', ['Manual Mastery Input','Simulate Student Sequence'], horizontal=True)
        mastery_vector = MasteryVector(graph, threshold)
        if mode=='Manual Mastery Input':
            st.markdown('Set your current mastery level for each topic:')
            cols=st.columns(2); nodes=list(graph.nodes)
            for i,node in enumerate(nodes):
                label=graph.nodes[node].get('label',node); level=graph.nodes[node].get('level','')
                val=cols[i%2].slider(f'{label} ({level})',0.0,1.0,0.0,0.05,key=f'mastery_{node}')
                mastery_vector.update(node,val)
        else:
            seq_length=st.slider('Sequence Length',10,200,50)
            seed=st.number_input('Student Seed',1,1000,42,1)
            np.random.seed(int(seed))
            sim_skills=np.random.randint(0,config['num_skills'],seq_length).tolist()
            sim_corrects=np.random.randint(0,2,seq_length).tolist()
            skill_probs=run_sakt_inference(model,config,sim_skills,sim_corrects,device)
            mastery_vector=build_mastery_vector(skill_probs,graph,skill_encoder,domain_key,threshold)
            st.success(f'SAKT inference complete — {len(skill_probs)} skill predictions generated')
        if st.button('🚀 Generate Recommendations', type='primary'):
            output=pipeline.run(mastery_vector)
            summary=mastery_vector.get_mastery_summary()
            col1,col2,col3,col4=st.columns(4)
            col1.metric('Topics Mastered',f"{summary['mastered']} / {summary['total_topics']}")
            col2.metric('Mastery Rate',f"{summary['mastery_rate']:.1%}")
            col3.metric('Approved Topics',output['total_approved'])
            col4.metric('Violation Rate',f"{output['prerequisite_violation_rate']:.1%}")
            st.markdown('---')
            st.subheader(f'Top {top_n} Recommendations')
            if not output['top_recommendations']: st.warning('No recommendations — adjust mastery or lower threshold.')
            else:
                for i,rec in enumerate(output['top_recommendations'],1):
                    with st.expander(f"{i}. {rec['topic_label']} — Score: {rec['score']} | Mastery: {rec['mastery']:.1%}", expanded=(i<=3)):
                        st.markdown(f"**Reasoning:** {rec['reasoning']}")
                        st.progress(rec['mastery'])
            if output['vetoed_sample']:
                st.markdown('---'); st.subheader('⛔ Sample Vetoed Topics')
                for rec in output['vetoed_sample']:
                    with st.expander(f"✗ {rec['topic_label']}"):
                        st.markdown(f"**Reason:** {rec['reasoning']}")
    with tab2:
        st.header(f'{domain} Knowledge Map')
        st.markdown(f"**{graph.number_of_nodes()} topics** | **{graph.number_of_edges()} prerequisite relationships**")
        rows=[]
        for node in graph.nodes:
            label=graph.nodes[node].get('label',node); level=graph.nodes[node].get('level',''); term=graph.nodes[node].get('term','')
            prereqs=[graph.nodes[p].get('label',p) for p in graph.predecessors(node)]
            rows.append({'Topic':label,'Level':level,'Term':term,'Prerequisites':', '.join(prereqs) if prereqs else 'None (Foundational)'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        longest=nx.dag_longest_path(graph)
        st.markdown('**Longest prerequisite chain:**')
        st.markdown(' → '.join([graph.nodes[n].get('label',n) for n in longest]))
    with tab3:
        st.header('System Diagnostics')
        col1,col2=st.columns(2)
        with col1: st.subheader('Model Configuration'); st.json(config)
        with col2:
            st.subheader('DAG Statistics')
            st.json({'domain':domain,'nodes':graph.number_of_nodes(),'edges':graph.number_of_edges(),'is_valid_dag':nx.is_directed_acyclic_graph(graph),'longest_path':len(nx.dag_longest_path(graph))})
        st.subheader('Domain Switching')
        dcol1,dcol2=st.columns(2)
        with dcol1: st.metric('Math DAG',f'{math_graph.number_of_nodes()} topics')
        with dcol2: st.metric('CS DAG',f'{cs_graph.number_of_nodes()} topics')

if __name__ == '__main__':
    main()