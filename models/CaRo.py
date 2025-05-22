import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import TransformerEncoder
from collections import OrderedDict
from models.CrossAttention import Cross_Attn
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
eps = 1e-9


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)


def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike)
    return negLogLike


infonce_criterion = nn.CrossEntropyLoss()


def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)


class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen, self.ic_graph = raw_graph
        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(
            np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(
            np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = nn.functional.normalize(
            self.content_feature, dim=-1)
        self.text_feature = nn.functional.normalize(self.text_feature, dim=-1)

        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))

            for m in module:
                init(m)
            return module
        self.cross_attn = Cross_Attn()
        # encoders for media feature
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)

        self.multimodal_feature_dim = self.embedding_size
        # MM <<<

        # BI >>>
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_embeddings)
        self.multimodal_feature_dim += self.embedding_size
        # BI <<<

        # UI >>>
        self.cf_transformation = nn.Linear(
            self.embedding_size, self.embedding_size)
        init(self.cf_transformation)
        items_in_cf = self.ui_graph.sum(axis=0, dtype=bool)
        self.warm_indices_cf = torch.LongTensor(
            np.argwhere(items_in_cf)[:, 1]).to(device)
        self.cold_indices_cf = torch.LongTensor(
            np.argwhere(~items_in_cf)[:, 1]).to(device)
        self.multimodal_feature_dim += self.embedding_size
        # UI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size,
                             self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y
    
    def cross_attention(self, query, key, value):
        q = self.w_q(query)
        k = self.w_k(key)
        v =self.w_v(value)

        attn = (q@ k.transpose(-1,-2))*(self.embedding_size ** -0.5)
        attn = attn.softmax(dim = -1)

        output = attn@ v 

        output = output.mean(dim=-2)
        return output

    def forward_all(self):
        
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features_output, feature_cross = self.cross_attn(t_feature, c_feature, cf_feature_full)
        features_output = torch.split(features_output, 64, dim = 1)
        features_output = torch.stack(features_output, dim=1) 

        final_feature = self.selfAttention(F.normalize(features_output, dim=-1))
        # multimodal fusion <<< 

        return final_feature


    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        features = [mm_feature]
        bi_feature_full = self.item_embeddings
        bi_feature = bi_feature_full[seq_modify]
        features.append(bi_feature)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        cf_feature = cf_feature_full[seq_modify]
        features.append(cf_feature)

        features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]
        bs, n_token, N_modal, d = features.shape

        # multimodal fusion >>>
        final_feature = self.selfAttention(
            F.normalize(features.view(-1, N_modal, d), dim=-1))
        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        return final_feature

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        features = [mm_feature_full]

        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]
        size = features.shape[:2]  # (bs, #modality)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio  # the remainders are true
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)

            # multimodal fusion >>>
            final_feature = self.selfAttention(
                F.normalize(masked_feat, dim=-1))
            # multimodal fusion <<<
            return final_feature

        return random_mask(), random_mask()

class LightGCN(nn.Module):
    def __init__(self, num_items, num_cates, embedding_size, ic_graph, n_layers = 2, device = 'cuda'):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.n_layers = n_layers
        self.num_items = num_items
        self.num_cates = num_cates
        self.init_emb()
        
    def init_emb(self):
        self.cate_embedding = nn.Parameter(torch.FloatTensor(self.num_cates, self.embedding_size)).to(self.device)
        nn.init.xavier_normal_(self.cate_embedding)
        self.item_embedding = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size)).to(self.device)
        nn.init.xavier_normal_(self.item_embedding)
    def build_symmetric_adj(self, ic_graph, num_items, num_cates):
        A_ic = ic_graph.T
        A_ci = ic_graph
        
        #symmetric adjacency
        top = sp.hstack[sp.scr_matrix((num_items,num_items)), A_ic]
        bottom = sp.hstack[A_ci,sp.scr_matrix((num_cates,num_cates))]
        A = sp.vstack([top, bottom])

        #normalize
        row_sum = np.array(A.sum(1)).flatten()
        d_inv_sqrt =np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

    def forward(self):
        all_embeddings = torch.cat([self.item_embedding, self.cate_embedding], dim = 0)
        embeddings = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings =  torch.sparse.mm(self.graph, all_embeddings)
            embeddings.append(all_embeddings)
        final_emb = torch.stack(embeddings, dim = 0).mean(dim = 0)
        return final_emb[:self.item_embedding.shape[0]]

class CaRo(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.num_cate  = self.conf["num_cates"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen, self.ic_graph = raw_graph
        self.item_augmentation = self.conf["item_augment"]

        self.encoder = HierachicalEncoder(conf, raw_graph, features)
        # decoder has the similar structure of the encoder
        self.decoder = HierachicalEncoder(conf, raw_graph, features)

        self.bundle_encode = TransformerEncoder(conf={
            "n_layer": conf["trans_layer"],
            "dim": 64,
            "num_token": 100,
            "device": self.device,
        }, data={"sp_graph": self.bi_graph_seen})

        self.cl_temp = conf['cl_temp']
        self.cl_alpha = conf['cl_alpha']

        self.bundle_cl_temp = conf['bundle_cl_temp']
        self.bundle_cl_alpha = conf['bundle_cl_alpha']
        self.cl_projector = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cl_projector)
        if self.item_augmentation in ["FD", "MD"]:
            self.dropout_rate = conf["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.item_augmentation in ["FN"]:
            self.noise_weight = conf['noise_weight']

        self.get_cate_embbed(False)
        dense_ic = self.convert_sparse(self.ic_graph)
        self.item_cate_feat = dense_ic @ self.cate_feature
        self.lightgcn = LightGCN(num_items=self.num_item,num_cates = self.num_cate, embedding_size= self.embedding_size, ic_graph = self.ic_graph, device = self.device )

    def convert_sparse(self, sparse):
        dense_mat = sparse.toarray()
        dense_tensor= torch.tensor(dense_mat)
        return dense_tensor.to(self.device)
    def init_emb(self):
        self.cate_feature = nn.Parameter(torch.FloatTensor(self.num_cate, self.embedding_size)).to(self.device)
        nn.init.xavier_normal_(self.cate_feature)
    def get_cate_embbed(self, co_oc = False):
        dataset_name = 'pog'
        path = self.conf['data_path']
        if co_oc == True:
            cbc_cooc = sp.load_npz(f'{path}/{dataset_name}/cbc_cooc.npz')
            svd = TruncatedSVD(n_components=self.embedding_size)
            cate_embeddings = svd.fit_transform(cbc_cooc) 
            cate_embeddings_tensor = torch.FloatTensor(cate_embeddings).to(self.device)
            print(cate_embeddings.shape)
            self.cate_feature = cate_embeddings_tensor
            print("Done creating c_embed from cooc matrix")
        else:
            self.init_emb()
            # print(self.item_cate_feat.device)
            print("Random initialize c_embed")

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        feat_bundle_view = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view = self.decoder(batch, all=True)

        # compute loss >>>
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)

        #item cate lightgcn
        item_features_cate = self.lightgcn()
        prediction = bundle_feature@item_features_cate.T
        logits += prediction 

        loss = recon_loss_function(logits, full)  # main_loss

        # # item-level contrastive learning >>>
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        item_loss = torch.tensor(0).to(self.device)
        if self.cl_alpha > 0:
            if self.item_augmentation == "FN":
                item_features = (self.encoder(batch, all=True) + self.item_cate_feat)[items_in_batch]
                sub1 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                sub2 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features)
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), sub2.view(-1, self.embedding_size), self.cl_temp)
        # # item-level contrastive learning <<<

        # bundle-level contrastive learning >>>
        bundle_loss = torch.tensor(0).to(self.device)
        if self.bundle_cl_alpha > 0:
            feat_bundle_view2 = self.encoder(seq_modify)  # [bs, n_token, d]
            bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
            bundle_loss = self.bundle_cl_alpha * cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), bundle_feature2.view(-1, self.embedding_size), self.bundle_cl_temp)
        # bundle-level contrastive learning <<<

        return {
            'loss': loss + item_loss + bundle_loss,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach()
        }


    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view = self.encoder(seq_x)

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view = self.decoder(
            (idx, x, seq_x, None, None), all=True)
        item_feature_cate = self.lightgcn()
        prediction = bundle_feature@item_feature_cate.T 
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        logits += prediction
        return logits

    def propagate(self, test=False):
        return None