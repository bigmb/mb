## file to view pca / umap / tsne embeddings in 2d or 3d with tf projector and plotly

from mb import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import umap
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import numpy as np

__all__ = ['get_emb','viz_emb','generate_sprite_images']


def get_emb(df: pd.DataFrame, emb= 'embeddings', emb_type='umap', dim=2,keep_original_emb=False,file_save=None, logger=None,**kwargs):
    """
    Visualize embeddings in 2d or 3d with tf projector and plotly

    Args:
        df (pd.DataFrame): dataframe containing embeddings. File location or DataFrame object.
        emb (str): name of embedding column
        emb_type (str, optional): embedding type. Defaults to 'umap'.
        dim (int, optional): embedding dimension. Defaults to 2.
        keep_original_emb (bool, optional): keep original embedding column. Defaults to False.
        file_save (str, optional): file location to save embeddings csv. Defaults to None.
    Output:
        df (pd.DataFrame): dataframe containing embeddings. Original embedding column is dropped.
    """
    
    if type(df) is not pd.DataFrame:
        if logger:
            logger.info('Type of df :{}'.format(str(type(df))))
        df = pd.load_any_df(df)
        if logger:
            logger.info('Loaded dataframe from path {}'.format(str(df)))
    
    if logger:
        logger.info('Data shape {}'.format(str(df.shape)))
        logger.info('Data columns {}'.format(str(df.columns)))
        logger.info('Performing {} on {} embeddings'.format(emb_type,emb))
    
    if emb_type=='pca':
        pca = PCA(n_components=dim)
        pca_emb = pca.fit_transform(list(df[emb]))
        if logger:
            logger.info('First PCA transform result : {}'.format(str(pca_emb[0])))
        temp_res = list(pca_emb)
    
    if emb_type=='tsne':
        tsne = TSNE(n_components=dim, verbose=1, perplexity=30, n_iter=250, **kwargs)
        df[emb] = df[emb].apply(lambda x: np.array(x))
        k1 = np.vstack(df[emb])
        tsne_emb = tsne.fit_transform(k1)
        if logger:
            logger.info('First TSNE transform result : {}'.format(str(tsne_emb[0])))
        temp_res = list(tsne_emb)
    
    if emb_type=='umap':
        umap_emb = umap.UMAP(n_neighbors=dim, min_dist=0.3, metric='correlation',**kwargs).fit_transform(list(df[emb]))
        if logger:
            logger.info('First UMAP transform result : {}'.format(str(umap_emb[0])))
        temp_res = list(umap_emb)
    
    df['emb_res'] = temp_res
    if keep_original_emb==False:
        df.drop(emb,axis=1,inplace=True)
        if logger:
            logger.info('Dropped original embedding column')
            
    if file_save:
        df.to_csv(file_save,index=False)
    else:
        df.to_csv('./emb_res.csv',index=False)
    
    return df

def viz_emb(df: pd.DataFrame, emb_column='emb_res' , target_column='taxcode', viz_type ='plt',limit = None,image_tb=None , file_save=None,
            dont_viz=False, logger=None):
    """
    Vizualize embeddings in 2d or 3d with tf projector and plotly
    
    Args:
        df (pd.DataFrame): dataframe containing embeddings. File location or DataFrame object.
        emb_column (str): name of embedding column
        target_column (str): name of target column. It can be used to color the embeddings. Defaults to 'taxcode'. Can be None too.
        viz_type (str, optional): visualization type: 'plt' or 'tf'. Defaults to 'plt'.
        limit (int, optional): limit number of data points to visualize. Takes random samples. Defaults to None.
        image_tb (str, optional): image location column to be used in tensorboard projector if want to create with images. Defaults to None.
        file_save (str, optional): file location to save plot. If viz_type='tf', then it wont be saved. Defaults to None.
        dont_viz (bool, optional): if True, then it wont visualize. Defaults to False.
        logger (logger, optional): logger object. Defaults to None.
    Output:
        None
    """
    
    if type(df) != pd.DataFrame:
        if logger:
            logger.info('Type of df :{}'.format(str(type(df))))
        df = pd.load_any_df(df)
    
    if limit:
        df = df.sample(limit)
    
    assert emb_column in df.columns, 'Embedding column not found in dataframe'
    
    emb_data = np.concatenate(np.array(df[emb_column]))
    emb_data = emb_data.reshape(-1,2) #change this for 3d
    if logger:
        logger.info('Embedding data shape {}'.format(str(emb_data.shape)))
    
    if target_column:
        target_data = list(df[target_column])
        if type(target_data[0]) == str:
            target_data = LabelEncoder().fit_transform(target_data)
        
    assert target_column==None or target_column in df.columns, 'Target column not found in dataframe'
    
    if file_save == None:
        file_save = './emb_plot.png'
        
    # Visualize the embeddings using a scatter plot
    if viz_type=='plt' and target_column:
        plt.scatter(emb_data[:, 0], emb_data[:, 1], c=target_data, cmap='viridis')
        if dont_viz==False:
            plt.show()
        if file_save:
            plt.savefig(file_save)

    elif viz_type=='plt' and target_column==None:
        plt.scatter(emb_data[:, 0], emb_data[:, 1])
        if dont_viz==False:
            plt.show()
        if file_save:
            plt.savefig(file_save)

    elif viz_type=='tf' and target_column:
        
        ##check from here
        log_dir = './tp_logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        emb_data = np.array(emb_data)
        loc_emb_data = os.path.join(log_dir,'emb_data_tf.tsv')
        np.savetxt(loc_emb_data, emb_data, delimiter='\t')
        
        target_data = np.array(target_data)
        loc_target_data = os.path.join(log_dir,'labels_tf.tsv')
        np.savetxt(loc_target_data,target_data,delimiter='\t')
        
        if image_tb is not None:
            loc_sprite_image = os.path.join(log_dir,'sprite_image.png')
            generate_sprite_images(df[image_tb], file_save=loc_sprite_image, img_size=28 ,logger=None)        
        
        from tensorboard.plugins import projector
        
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_path = loc_emb_data
        embedding.metadata_path = loc_target_data
        if image_tb is not None:
            embedding.sprite.image_path = loc_sprite_image
            embedding.sprite.single_image_dim.extend([32, 32])

        with open(os.path.join(log_dir, 'projector_config.pbtxt'), 'w') as f:
            f.write(str(config))
        
        if logger:
            logger.info('Run tensorboard --logdir={} to view embeddings'.format(log_dir))
            logger.info('if on jupyter notebook, run below code to view embeddings in notebook')
            logger.info('%load_ext tensorboard')
            logger.info('%tensorboard --logdir={}'.format(log_dir))

    
def generate_sprite_images(img_paths, file_save=None, img_size= 28 ,logger=None):
    """
    Create a sprite image consisting of images

    Args:
        img_paths (list or pd.DataFrame): list of image paths
        file_save (str, optional): file location to save sprite image. Defaults to None. Will save in current directory.
        img_size (int, optional): image size. Defaults to 28.
        logger (logger, optional): logger object. Defaults to None.
    Output:
        sprite_image (np.array): sprite image
    """
    
    if type(img_paths) is not list:
        img_paths = list(img_paths)
    
    #create sprite image
    images = [tf.io.read_file(img_path) for img_path in img_paths]
    images = [tf.image.decode_image(img) for img in images]
    images = [tf.image.resize(img, (img_size, img_size)) for img in images]
    images = [img.numpy() for img in images]
    sprite_image = np.concatenate(images, axis=1)
    
    if file_save:
        sprite_image.save(file_save)
    else:
        sprite_image.save('./sprite_image.png')
        
    return sprite_image
    