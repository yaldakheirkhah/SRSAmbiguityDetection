"""Conservative, deterministic partial recovery. Never reads data/splits.

Only raw Text/content identity is recoverable; candidate edges are co-isolation
constraints, not assertions of augmentation parentage. All nonmatched rows stay
quarantined. The protocol must already exist and is never outcome-tuned here.
"""

# Historical assertions are integrity controls; never disable them.
if not __debug__:
    raise RuntimeError("Optimized Python is prohibited for this research workflow")

from pathlib import Path
import collections
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as l2normalize

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
LABELS = [1, 2, 3, 4, 5, 6]


def read_authorized(path, **kwargs):
    resolved = (ROOT / path).resolve()
    if resolved not in {(ROOT / 'data/raw/dataset.csv').resolve(), (ROOT / 'final.csv').resolve()}:
        raise ValueError('Dataset read is not allowlisted: ' + str(resolved))
    return pd.read_csv(resolved, **kwargs)


def sha(path):
    p = Path(path).resolve()
    if (ROOT / 'data/splits').resolve() in p.parents or p.name.lower() == 'test.csv':
        raise ValueError('Protected path; hashing forbidden')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_key(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip().casefold()


def punctuation_key(key):
    # Preserve word/number token boundaries and comparison/arithmetic operators.
    # This is only candidate grouping, never a semantic equivalence assertion.
    return ' '.join(re.findall(r'\w+|[<>=+/%#@&|^~]', key, flags=re.UNICODE))


def tokens(key):
    return set(re.findall(r'\w+', key, flags=re.UNICODE))


def digest_id(prefix, value):
    return prefix + hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def bins(values):
    values = np.asarray(values, dtype=int)
    bounds = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None)]
    return {str(lo) if lo == hi else (f'{lo}-{hi}' if hi else f'{lo}+'):
            int(((values >= lo) & ((values <= hi) if hi is not None else True)).sum()) for lo, hi in bounds}


def main():
    protocol_path = BASE / 'config/protocol.json'
    cfg = json.loads(protocol_path.read_text())
    expected_protocol_hash = (BASE / 'config/protocol.sha256').read_text().split()[0]
    assert sha(protocol_path) == expected_protocol_hash, 'Predeclared protocol changed'
    if (BASE / 'reports/QUALITY_GATE.json').exists():
        raise SystemExit('Existing completed construction found. Refuse regeneration or overwrite.')
    assert cfg['eligible_for_original_frozen_test'] is False
    # Empty generated directories are absent from a fresh Git clone.
    for name in ['audit', 'data', 'reports']:
        directory = BASE / name
        if directory.is_symlink():
            raise ValueError('Generated directories may not be symlinks')
        directory.mkdir(exist_ok=True)
    print('STAGE 1: authorized sources and exact identity', flush=True)
    raw = read_authorized('data/raw/dataset.csv', encoding='latin-1')
    corpus = read_authorized('final.csv')
    assert list(raw.columns) == ['Text', 'Class']
    assert list(corpus.columns) == ['Text', 'Class', 'contraction_count', 'text']
    assert len(raw) == 7061 and len(corpus) == 24949
    assert raw[['Text', 'Class']].notna().all().all() and corpus[['Text', 'Class']].notna().all().all()
    assert set(raw.Class) == set(LABELS) and set(corpus.Class) == set(LABELS)
    raw_keys, corpus_keys = raw.Text.map(content_key), corpus.Text.map(content_key)
    assert raw_keys.ne('').all() and corpus_keys.ne('').all()
    keys = sorted(set(raw_keys) | set(corpus_keys))
    key_to_node = {key: i for i, key in enumerate(keys)}
    node_ids = [digest_id('text_', k) for k in keys]
    assert len(set(node_ids)) == len(keys)
    raw_nodes = raw_keys.map(key_to_node).to_numpy(dtype=int)
    corpus_nodes = corpus_keys.map(key_to_node).to_numpy(dtype=int)
    raw_members, final_members = collections.defaultdict(list), collections.defaultdict(list)
    node_labels = collections.defaultdict(set)
    for i, node in enumerate(raw_nodes):
        raw_members[node].append(i); node_labels[node].add(int(raw.Class.iloc[i]))
    for i, node in enumerate(corpus_nodes):
        final_members[node].append(i); node_labels[node].add(int(corpus.Class.iloc[i]))
    matched = np.array([n in raw_members for n in corpus_nodes])
    assert int(matched.sum()) == 7930 and int((~matched).sum()) == 17019
    assert int(corpus.duplicated().sum()) == 3082
    uf = UnionFind(len(keys))
    punct_groups = collections.defaultdict(list)
    punct_keys = [punctuation_key(k) for k in keys]
    for i, key in enumerate(punct_keys):
        if key: punct_groups[key].append(i)
    graph_edges = []
    for members in punct_groups.values():
        for j in members[1:]:
            uf.union(members[0], j)
            graph_edges.append((members[0], j, 'punctuation_variant', None, None))
    token_sets = [tokens(k) for k in keys]
    print(f'Unique union Text nodes: {len(keys)}; final unique={len(set(corpus_nodes))}; punctuation edges={len(graph_edges)}', flush=True)
    print('STAGE 2: full-corpus character candidate graph (construction only, not classifier fitting)', flush=True)
    c = cfg['character_vectorizer']
    cv = TfidfVectorizer(analyzer=c['analyzer'], ngram_range=tuple(c['ngram_range']), min_df=c['min_df'], dtype=np.float32)
    X = cv.fit_transform(keys)
    print(f'Character matrix {X.shape}, nnz={X.nnz}', flush=True)
    strong, risk = cfg['strong_lexical_candidate'], cfg['cross_family_lexical_risk']
    lexical_edges = []
    raw_cosine_pairs = 0
    for start in range(0, len(keys), 96):
        product = (X[start:start+96] @ X.T).tocoo()
        mask = (product.data >= risk['char_tfidf_cosine_min']) & (product.col > product.row + start)
        for ii, j, value in zip(product.row[mask], product.col[mask], product.data[mask]):
            i, j = int(ii + start), int(j)
            raw_cosine_pairs += 1
            if min(len(keys[i]), len(keys[j])) < risk['minimum_characters'] or min(len(token_sets[i]), len(token_sets[j])) < risk['minimum_tokens']:
                continue
            jac = len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j])
            if jac < risk['token_jaccard_min']: continue
            value = float(min(1.0, value))
            is_strong = value >= strong['char_tfidf_cosine_min'] and jac >= strong['token_jaccard_min']
            kind = 'strong_near_duplicate_candidate' if is_strong else 'looser_cross_family_risk'
            lexical_edges.append((i, j, kind, value, jac))
            if is_strong:
                uf.union(i, j); graph_edges.append((i, j, kind, value, jac))
        if start % (96 * 25) == 0:
            print(f'Character scan {start}/{len(keys)}; retained candidate edges={len(lexical_edges)}', flush=True)
    components = collections.defaultdict(list)
    for i in range(len(keys)): components[uf.find(i)].append(i)
    group_for_node, groups = {}, {}
    for members in components.values():
        gid = digest_id('family_', '\n'.join(node_ids[i] for i in members))
        assert gid not in groups
        for node in members: group_for_node[node] = gid
        anchors = [node for node in members if node in raw_members]
        final_indices = [r for node in members for r in final_members[node]]
        labels = sorted(set().union(*(node_labels[node] for node in members)))
        reasons = []
        q = cfg['family_quarantine']
        if len(labels) != 1: reasons.append('cross_label_conflict')
        if len(members) > q['maximum_unique_text_nodes'] or len(final_indices) > q['maximum_final_rows']:
            reasons.append('oversized_cluster')
        if len(anchors) > q['maximum_distinct_original_normalized_anchors']:
            reasons.append('uncertain_multiple_original_anchors')
        groups[gid] = {'members':members,'anchors':anchors,'final_indices':final_indices,'labels':labels,'reasons':reasons}
    assert len(set(group_for_node.values())) == len(groups)
    cross_risk_edges = []
    for edge in lexical_edges:
        i, j = edge[:2]; ga, gb = group_for_node[i], group_for_node[j]
        if ga != gb:
            cross_risk_edges.append(edge)
            for gid in [ga, gb]:
                if 'unresolved_cross_family_lexical_risk' not in groups[gid]['reasons']:
                    groups[gid]['reasons'].append('unresolved_cross_family_lexical_risk')
    print(f'Candidate components={len(groups)}, cross-family looser risk edges={len(cross_risk_edges)}', flush=True)
    print('STAGE 3: separate latent lexical-semantic candidate investigation; no verified parent assignments', flush=True)
    sem = cfg['semantic_review_retrieval']
    wv = TfidfVectorizer(ngram_range=tuple(sem['word_ngram_range']), min_df=sem['min_df'], max_features=sem['max_features'], dtype=np.float32)
    words = wv.fit_transform(keys)
    components_n = min(sem['components'], words.shape[1]-1, words.shape[0]-1)
    svd = TruncatedSVD(n_components=components_n, random_state=sem['random_state'], n_iter=5)
    latent = l2normalize(svd.fit_transform(words)).astype(np.float32)
    semantic_edges, seen_semantic = [], set()
    for start in range(0, len(keys), 128):
        scores = latent[start:start+128] @ latent.T
        for offset in range(scores.shape[0]): scores[offset, start+offset] = -np.inf
        top = np.argpartition(scores, -sem['nearest_neighbors_per_node'], axis=1)[:, -sem['nearest_neighbors_per_node']:]
        for offset, candidates in enumerate(top):
            i = start + offset
            for j in candidates:
                j = int(j); score = float(scores[offset,j])
                pair = (min(i,j), max(i,j))
                if score < sem['latent_cosine_min'] or pair in seen_semantic: continue
                seen_semantic.add(pair)
                if group_for_node[i] == group_for_node[j]: continue
                jac = len(token_sets[i] & token_sets[j]) / max(1,len(token_sets[i] | token_sets[j]))
                if jac >= sem['token_jaccard_max_exclusive']: continue
                char_score = float(X[i].multiply(X[j]).sum())
                if char_score >= sem['char_cosine_max_exclusive']: continue
                semantic_edges.append((i,j,'latent_semantic_review_candidate',score,jac,char_score))
                for gid in [group_for_node[i],group_for_node[j]]:
                    if 'unreviewed_latent_semantic_candidate' not in groups[gid]['reasons']:
                        groups[gid]['reasons'].append('unreviewed_latent_semantic_candidate')
        if start % (128*40) == 0: print(f'Latent scan {start}/{len(keys)}, review pairs={len(semantic_edges)}', flush=True)
    print(f'Separate semantic review candidates={len(semantic_edges)}; this is not semantic proof or exhaustive recall', flush=True)
    audit = BASE / 'audit'
    def edge_frame(edges, semantic=False):
        rows = []
        for edge in edges:
            i,j,kind,score,jac = edge[:5]
            row={'node_a':node_ids[i],'node_b':node_ids[j],'family_a':group_for_node[i],'family_b':group_for_node[j],
                 'kind':kind,'similarity':score,'token_jaccard':jac,'Text_a':keys[i],'Text_b':keys[j],
                 'labels_a':','.join(map(str,sorted(node_labels[i]))),'labels_b':','.join(map(str,sorted(node_labels[j])))}
            if semantic:row['char_cosine']=edge[5]
            rows.append(row)
        cols=['node_a','node_b','family_a','family_b','kind','similarity','token_jaccard','Text_a','Text_b','labels_a','labels_b']
        if semantic:cols.append('char_cosine')
        return pd.DataFrame(rows,columns=cols)
    edge_frame(graph_edges).to_csv(audit/'candidate_group_edges.csv',index=False)
    edge_frame(lexical_edges).to_csv(audit/'lexical_candidates.csv',index=False)
    edge_frame(cross_risk_edges).to_csv(audit/'unresolved_cross_family_lexical_edges.csv',index=False)
    edge_frame(semantic_edges,True).to_csv(audit/'semantic_review_candidates.csv',index=False)
    node_records=[]
    for i,key in enumerate(keys):
        node_records.append({'node_id':node_ids[i],'group_id':group_for_node[i],'normalized_Text':key,'punctuation_key':punct_keys[i],
                             'raw_record_ids':json.dumps([f'raw_{r:06d}' for r in raw_members[i]]),
                             'raw_labels':','.join(map(str,sorted(set(int(raw.Class.iloc[r]) for r in raw_members[i])))),
                             'legacy_row_ids':json.dumps([f'legacy_{r:06d}' for r in final_members[i]]),
                             'all_labels':','.join(map(str,sorted(node_labels[i])))})
    pd.DataFrame(node_records).to_csv(audit/'normalized_text_nodes.csv',index=False)
    group_records=[]
    for gid,g in sorted(groups.items()):
        group_records.append({'group_id':gid,'unique_text_nodes':len(g['members']),'legacy_rows':len(g['final_indices']),
                              'raw_original_records':sum(len(raw_members[n]) for n in g['anchors']),
                              'distinct_original_normalized_anchors':len(g['anchors']),
                              'labels':','.join(map(str,g['labels'])),'has_cross_label_conflict':len(g['labels'])>1,
                              'quarantine_reasons':';'.join(g['reasons']),
                              'anchor_node_ids':json.dumps([node_ids[n] for n in g['anchors']]),
                              'status':'inferred co-isolation component; NOT verified augmentation family'})
    gf=pd.DataFrame(group_records);gf.to_csv(audit/'candidate_families.csv',index=False)
    print('STAGE 4: partial recovery, quarantine and deterministic group allocation', flush=True)
    first_full = {}
    duplicate_of = []
    for i,row in corpus.iterrows():
        record=tuple(None if pd.isna(v) else v for v in row.tolist())
        previous=first_full.setdefault(record,i)
        duplicate_of.append('' if previous==i else f'legacy_{previous:06d}')
    representatives={};lineage=[];model_rows=[]
    eligible_source_rows=0;reason_counts=collections.Counter()
    for i,node in enumerate(corpus_nodes):
        node=int(node);gid=group_for_node[node];g=groups[gid]
        reasons=list(g['reasons'])
        if not matched[i]:reasons.insert(0,'no_verified_original_content_match')
        raw_label_set={int(raw.Class.iloc[r]) for r in raw_members[node]}
        if matched[i] and raw_label_set!={int(corpus.Class.iloc[i])}:
            reasons.append('original_label_disagreement_or_conflict')
        if reasons:
            disposition='quarantined'
            reason_counts.update(set(reasons))
        else:
            eligible_source_rows+=1
            if node in representatives:
                disposition='excluded_repeated_normalized_original_text'
            else:
                representatives[node]=i;disposition='retained_model_example'
                model_rows.append({'row_id':f'legacy_{i:06d}','group_id':gid,'Text':corpus.Text.iloc[i],
                                   'Class':int(corpus.Class.iloc[i]),'normalized_node_id':node_ids[node],
                                   'origin':'normalized_original_content_match_only','original_frozen_test_eligible':False})
        lineage.append({'row_id':f'legacy_{i:06d}','csv_zero_based_row_index':i,'Class':int(corpus.Class.iloc[i]),
                        'normalized_node_id':node_ids[node],'group_id':gid,
                        'normalized_original_content_match':bool(matched[i]),
                        'matching_raw_record_ids':json.dumps([f'raw_{r:06d}' for r in raw_members[node]]),
                        'matching_raw_labels':','.join(map(str,sorted(raw_label_set))),
                        'verified_augmentation_parent_id':'',
                        'parentage_status':'content_identity_only_not_causal_parentage' if matched[i] else 'unverified',
                        'candidate_original_anchor_count':len(g['anchors']),'candidate_family_final_rows':len(g['final_indices']),
                        'candidate_family_unique_texts':len(g['members']),'full_duplicate_of_row_id':duplicate_of[i],
                        'disposition':disposition,'quarantine_reasons':';'.join(reasons)})
    lineage=pd.DataFrame(lineage)
    model=pd.DataFrame(model_rows)
    lineage.to_csv(audit/'lineage_group_table.csv',index=False)
    lineage.loc[lineage.disposition=='quarantined'].to_csv(audit/'quarantined_rows.csv',index=False)
    corpus_class_counts={str(k):int(v) for k,v in corpus.Class.value_counts().sort_index().items()}
    final_groups=gf.loc[gf.legacy_rows>0]
    summary={
        'benchmark_id':cfg['benchmark_id'],'protocol_sha256':expected_protocol_hash,
        'sources':{'raw':{'path':'data/raw/dataset.csv','rows':len(raw),'sha256':sha(ROOT/'data/raw/dataset.csv')},
                   'legacy':{'path':'final.csv','rows':len(corpus),'sha256':sha(ROOT/'final.csv')}},
        'class_counts_raw':{str(k):int(v) for k,v in raw.Class.value_counts().sort_index().items()},
        'class_counts_legacy':corpus_class_counts,'full_duplicate_rows':int(corpus.duplicated().sum()),
        'normalized_duplicate_legacy_rows':int(corpus_keys.duplicated().sum()),
        'normalized_original_matching_rows':int(matched.sum()),'unlinked_rows':int((~matched).sum()),
        'raw_unique_normalized_texts':int(raw_keys.nunique()),'legacy_unique_normalized_texts':int(corpus_keys.nunique()),
        'missing_derived_text_rows_unused':int(corpus.text.isna().sum()),
        'candidate_families_with_legacy_rows':len(final_groups),'candidate_families_all_raw_plus_legacy':len(groups),
        'family_legacy_row_size_bins':bins(final_groups.legacy_rows),'family_unique_text_size_bins':bins(final_groups.unique_text_nodes),
        'family_legacy_row_size_quantiles':{str(q):float(final_groups.legacy_rows.quantile(q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        'families_with_cross_label_conflicts':int(final_groups.has_cross_label_conflict.sum()),
        'families_oversized':int(final_groups.quarantine_reasons.str.contains('oversized_cluster').sum()),
        'families_with_multiple_original_anchors':int((final_groups.distinct_original_normalized_anchors>1).sum()),
        'candidate_graph_edges':len(graph_edges),'strong_near_duplicate_edges':sum(e[2]=='strong_near_duplicate_candidate' for e in graph_edges),
        'punctuation_edges':sum(e[2]=='punctuation_variant' for e in graph_edges),
        'char_cosine_pairs_over_screening_threshold':raw_cosine_pairs,'lexical_candidates':len(lexical_edges),
        'unresolved_cross_family_lexical_edges_before_quarantine':len(cross_risk_edges),
        'semantic_review_candidates':len(semantic_edges),'semantic_paraphrase_recall':'UNKNOWN; local latent retrieval is not a validated semantic model',
        'lsa_variance_ratio_sum':float(svd.explained_variance_ratio_.sum()),
        'eligible_source_rows_before_deduplication':eligible_source_rows,
        'retained_model_examples':len(model),'quarantined_source_rows':int((lineage.disposition=='quarantined').sum()),
        'excluded_repeated_original_text_rows':int((lineage.disposition=='excluded_repeated_normalized_original_text').sum()),
        'quarantine_reason_row_counts_overlapping':dict(reason_counts),
        'retained_class_counts':{} if model.empty else {str(k):int(v) for k,v in model.Class.value_counts().sort_index().items()},
        'unlinked_rows_by_candidate_original_anchor_count':{
            'zero':int(((lineage.candidate_original_anchor_count==0)&~lineage.normalized_original_content_match).sum()),
            'one':int(((lineage.candidate_original_anchor_count==1)&~lineage.normalized_original_content_match).sum()),
            'multiple':int(((lineage.candidate_original_anchor_count>1)&~lineage.normalized_original_content_match).sum())},
        'environment':{'python':sys.version,**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy']}},
        'full_corpus_grouping_status':'FAIL_VERIFIED_PARENTAGE; no full25k split produced',
        'scope':'Partial recovery of raw-matched original content only; does not test historical augmentation benefit',
        'eligible_for_original_frozen_test':False,'test_file_accessed':False,'clean_validation_file_accessed':False,
        'label_changes':0,'input_datasets_modified':False,
    }
    checks={'all_24949_rows_accounted_once':len(lineage)==24949 and lineage.row_id.is_unique,
            'all_unlinked_rows_quarantined':bool(lineage.loc[~lineage.normalized_original_content_match].disposition.eq('quarantined').all()),
            'retained_nonempty':not model.empty,
            'minimum_25_eligible_groups_each_class':not model.empty and set(model.Class)==set(LABELS) and model.groupby('Class').group_id.nunique().min()>=cfg['split']['minimum_eligible_groups_per_class']}
    train=None;valid=None
    if all(checks.values()):
        mg=model[['group_id','Class']].drop_duplicates().sort_values('group_id')
        assert mg.group_id.is_unique
        train_groups,valid_groups=train_test_split(mg.group_id.to_numpy(),test_size=cfg['split']['validation_fraction'],
                                                 random_state=cfg['split']['random_state'],stratify=mg.Class)
        train=model.loc[model.group_id.isin(set(train_groups))].sort_values('row_id').reset_index(drop=True)
        valid=model.loc[model.group_id.isin(set(valid_groups))].sort_values('row_id').reset_index(drop=True)
        partition={gid:'train' for gid in train_groups};partition.update({gid:'validation' for gid in valid_groups})
        model['partition']=model.group_id.map(partition)
        lineage['partition']=lineage.group_id.map(partition).fillna('quarantine_or_excluded')
        lineage.loc[lineage.disposition!='retained_model_example','partition']='quarantine_or_excluded'
        checks.update({
            'train_validation_group_overlap_zero':not set(train.group_id)&set(valid.group_id),
            'train_validation_normalized_text_overlap_zero':not set(train.Text.map(content_key))&set(valid.Text.map(content_key)),
            'train_validation_punctuation_key_overlap_zero':not set(train.Text.map(content_key).map(punctuation_key))&set(valid.Text.map(content_key).map(punctuation_key)),
            'every_retained_family_passes_quarantine_rules':all(not groups[g]['reasons'] for g in model.group_id),
            'all_retained_are_normalized_original_matches':bool(lineage.loc[lineage.disposition=='retained_model_example'].normalized_original_content_match.all()),
            'minimum_20_train_each_class':set(train.Class)==set(LABELS) and train.Class.value_counts().min()>=cfg['split']['minimum_training_examples_per_class'],
            'minimum_5_validation_each_class':set(valid.Class)==set(LABELS) and valid.Class.value_counts().min()>=cfg['split']['minimum_validation_examples_per_class'],
        })
        retained_groups=set(model.group_id)
        checks['retained_endpoints_of_cross_family_lexical_risk_zero']=not any(group_for_node[e[0]] in retained_groups or group_for_node[e[1]] in retained_groups for e in cross_risk_edges)
        checks['retained_endpoints_of_semantic_review_candidates_zero']=not any(group_for_node[e[0]] in retained_groups or group_for_node[e[1]] in retained_groups for e in semantic_edges)
        checks['no_cross_partition_strong_edge']=not any(partition.get(group_for_node[e[0]]) and partition.get(group_for_node[e[1]]) and partition[group_for_node[e[0]]]!=partition[group_for_node[e[1]]] for e in graph_edges)
        summary.update({'train_rows':len(train),'validation_rows':len(valid),'train_groups':int(train.group_id.nunique()),'validation_groups':int(valid.group_id.nunique()),
                        'train_class_counts':{str(k):int(val) for k,val in train.Class.value_counts().sort_index().items()},
                        'validation_class_counts':{str(k):int(val) for k,val in valid.Class.value_counts().sort_index().items()},
                        'retained_family_model_example_size_bins':bins(model.groupby('group_id').size())})
        model.to_csv(audit/'retained_model_examples.csv',index=False)
        lineage.to_csv(audit/'lineage_group_table.csv',index=False)
        # Determinism replay uses the same sorted group table and fixed settings, no evaluation metrics.
        t2,v2=train_test_split(mg.group_id.to_numpy(),test_size=cfg['split']['validation_fraction'],random_state=cfg['split']['random_state'],stratify=mg.Class)
        checks['deterministic_group_split_replay']=np.array_equal(t2,train_groups) and np.array_equal(v2,valid_groups)
    passed=all(bool(x) for x in checks.values())
    gate={'status':'PASS_PARTIAL_RECOVERY' if passed else 'FAIL_PARTIAL_RECOVERY',
          'approved_for_gpu_package':passed,'eligible_for_original_frozen_test':False,
          'full_25k_provenance_pass':False,'semantic_parentage_verified':False,'semantic_recall':'UNKNOWN',
          'benchmark_id':cfg['benchmark_id'],'protocol_sha256':expected_protocol_hash,'checks':{k:bool(x) for k,x in checks.items()},
          'scope':'Bounded textual-isolation checks for partial original-content recovery only; NOT leakage-free, NOT a full25k augmentation experiment',
          'limitations':cfg['scientific_limitations']}
    if passed:
        train.to_csv(BASE/'data/train.csv',index=False);valid.to_csv(BASE/'data/validation.csv',index=False)
        summary['new_partition_sha256']={name:sha(BASE/f'data/{name}.csv') for name in ['train','validation']}
    summary['quality_gate']=gate
    (BASE/'reports/CONSTRUCTION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    (BASE/'reports/QUALITY_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    counts=[]
    for label in LABELS:
        counts.append({'Class':label,'legacy_rows':int((corpus.Class==label).sum()),'normalized_raw_match_rows':int(((corpus.Class==label).to_numpy()&matched).sum()),
                       'eligible_source_rows_before_dedup':int(((lineage.Class==label)&lineage.disposition.isin(['retained_model_example','excluded_repeated_normalized_original_text'])).sum()),
                       'retained_examples':0 if model.empty else int((model.Class==label).sum()),
                       'train':0 if train is None else int((train.Class==label).sum()),'validation':0 if valid is None else int((valid.Class==label).sum())})
    pd.DataFrame(counts).to_csv(BASE/'reports/class_distributions.csv',index=False)
    print(json.dumps({k:summary[k] for k in ['full_corpus_grouping_status','candidate_families_with_legacy_rows','eligible_source_rows_before_deduplication','retained_model_examples','quarantined_source_rows','excluded_repeated_original_text_rows','retained_class_counts']},indent=2),flush=True)
    print('QUALITY_GATE',gate['status'],'failed checks',[k for k,val in checks.items() if not val],flush=True)
    if passed:print('NEW SPLIT',len(train),len(valid),flush=True)


if __name__=='__main__':
    main()
