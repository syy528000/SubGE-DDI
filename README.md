# SubGE-DDI
SubGE-DDI: a new drug-drug interaction prediction model based on biomedical texts and drug-pairs knowledge subgraph enhancement

Abstract
Biomedical texts are an important source of drug-drug interactions (DDIs) information in current pharmacovigilance. How to mine effective DDIs from biomedical texts and predict unkonwn DDIs from them is one of the hot spots in current research. However, the huge demand for accurate manual annotations has been the biggest factor hindering the performance improvement and application of machine learning algorithms. To overcome this problem, we propose a new DDI prediction framework named Subgraph Enhance model for DDI (SubGE-DDI) that uses drug pairs knowledge subgraph information to assist large-scale plain text prediction without many annotations. The drug pairs knowledge subgraph is derived from a huge drug knowledge graph, which consists of various public datasets, such as DrugBank, TwoSIDES, OffSIDES, DrugCentral, EntrezeGene, SMPDB (The Small Molecule Pathway Database), CTD (The Comparative Toxicogenomics Database) and SIDER. We evaluated SubGE-DDI on the public dataset (SemEval-2013 Task 9 dataset) and compared with other state-of-the-art baselines. The results showed that SubGE-DDI achieved an 83.91% micro F1 score and an 84.75% macro F1 score in the test dataset, which outperforms other state-of-the-art baselines. Our proposed drug pairs knowledge subgraph assisted model can effectively improve the performance of DDIs prediction from biomedical texts.
Keywords: DDIs prediction, biomedical text mining, knowledge graph

Run Code

To train&test the SubGE-DDI model:

python runDDIE.py --task_name MRPC --model_type bert --data_dir  G:/syy/code/share_code/data/data_train_test  --model_name_or_path  G:/syy/code/share_code/pubmedbert  --tokenizer_name  G:/syy/code/share_code/pubmedbert/vocab.txt  --output_dir  G:/syy/code/share_code/output --do_train  --num_train_epochs 5.  --dropout_prob .1  --weight_decay .01  --do_lower_case  --max_seq_length 390  --conv_window_size  5  --pos_emb_dim  10  --activation gelu  --per_gpu_train_batch_size  32 --work_dir G:/syy/code/share_code -d drugbank --hop 3 --graph_dim 75  --overwrite_output_dir  -k 5 --do_test --use_sub --use_cnn




