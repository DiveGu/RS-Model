"""
参数设置
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RS-Model.")
    # 路径参数
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='F:/data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='F:/data/experiment_output/',
                        help='Project path.')
    # 数据集 数据处理参数
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset from {lastfm,gowalla, yelp2018, amazon-book}')

    parser.add_argument('--prepro',nargs='?',default='5-core',
                        help='Choose data preprocess from {orgin,x-filter,x-core}')

    parser.add_argument('--test_method',nargs='?',default='ufo',
                        help='Choose a way to get test dataset from {fo, loo, tloo, tfo}')
    # 模型参数
    parser.add_argument('--model_type',nargs='?',default='bprmf',
                        help='Choose a model from {bprmf,neumf,DisenMF,LightGCN,NAIS,DGCF}.')
    parser.add_argument('--model_des',nargs='?',default='train_test',
                        help='record something')


    ## NeuMF 参数
    #parser.add_argument('--layers', nargs='?', default='[40,20]',
    #                    help='MLP sizes.')
    ## ---------------------------------------------------------

    ## DisenMF 参数
    #parser.add_argument('--factor_num', type=int,default=4,
    #                    help='factor num.')
    #parser.add_argument('--factor_dim', type=int,default=5,
    #                    help='factor num.')
    #parser.add_argument('--factor_class_layers', nargs='?', default='[5,5]',
    #                    help='factor class layers.')
    ## ---------------------------------------------------------

    ## LightGCN参数
    #parser.add_argument('--layers', nargs='?', default='[40,20]',
    #                    help='MLP sizes in NGCF.')
    #parser.add_argument('--layer_num', type=int,default=2,
    #                    help='layer_num in GCN.')
    ## ---------------------------------------------------------
    
    # DGCF参数
    parser.add_argument('--n_iteration', type=int,default=5,
                        help='iteration_num in dynamic_routing.')
    parser.add_argument('--layer_num', type=int,default=1,
                        help='layer_num in DGCF.')
    parser.add_argument('--factor_num', type=int,default=4,
                        help='factor num.')
    #parser.add_argument('--factor_dim', type=int,default=5,
    #                    help='factor num.')
    # ---------------------------------------------------------

    parser.add_argument('--embed_size',type=int,default=20,
                        help='CF embedding size')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-6,1e-8]',
                        help='Regularization.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='CF batch size.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Epoch number.')

    parser.add_argument('--verbose', type=int, default=10,
                        help='Display every verbose epoch.')

    # 模型是否需要读取【预训练参数】或者【保存的所有模型参数】
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2:Pretrain with stored models.')
    
    parser.add_argument('--pretrain_report', type=int, default=0,
                        help='show saved pretrained model preformance or not .. recall')
    # 是否需要保存模型的参数
    parser.add_argument('--save_model_flag', type=int, default=1,
                        help='save model parameters or not.')

    # 评价指标K
    parser.add_argument('--Ks', nargs='?', default='[1,5,10,20,50]',
                        help='top K.')
    parser.add_argument('--best_k_idx', type=int, default=2,
                        help='best recall k idx in Ks')
    parser.add_argument('--test_flag', nargs='?', default='all',
                        help='test rs part or all.')


    return parser.parse_args()

