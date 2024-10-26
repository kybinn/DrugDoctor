from sklearn import preprocessing
from torch.nn.functional import pad
from torch.utils import data
import torch
import random


class mimic_data(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




def pad_batch_v2_train(batch): # batch:长度为256的visit-level子集 
    batch_size = len(batch)

    # 统计当前batch中，每一个visit的疾病、手术、药物的最值
    d_max_num = 0           # 这个bath中，全部病人中，最大的diagnose数  ,37
    p_max_num = 0           # 这个bath中，全部病人中，最大的procedure数 ,18
    m_max_num = 0           # 这个bath中，全部病人中，最大的用药数      ,53
    
    # used_m_max_num = 0
    # used_visit_max_num = 0

    for data in batch: #data就是一个visit相关数据，包括diag，proc，med，used_med_list
        d_max_num = max(d_max_num, len(data[0]))
        p_max_num = max(p_max_num, len(data[1]))
        m_max_num = max(m_max_num, len(data[2]))
        # used_visit_max_num = max(used_visit_max_num, len(data[3])) 
        # for i in range(len(data[3])):
        #     used_m_max_num = max(used_m_max_num, len(data[3][i]))

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, d_max_num), -1e9)
    for i in range(batch_size):
        d_mask_matrix[i, :len(batch[i][0])] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, p_max_num), -1e9)
    for i in range(batch_size):
        p_mask_matrix[i, :len(batch[i][1])] = 0.

    # 生成 m_mask_matrix
    m_mask_matrix = torch.full((batch_size, m_max_num), -1e9)
    for i in range(batch_size):
        m_mask_matrix[i, :len(batch[i][2])] = 0.


    # 分别生成diagnoses、procedure、medication的数据
    diagnoses_tensor = torch.full((batch_size, d_max_num), -1)# torch.Size([256, 37])
    procedure_tensor = torch.full((batch_size, p_max_num), -1)#torch.Size([256, 18])
    medication_tensor = torch.full((batch_size, m_max_num), 0)#torch.Size([256, 53])
    # used_med_tensor = torch.full((batch_size, used_visit_max_num ,used_m_max_num), -1)#torch.Size([256, 60])
    used_med = []
    used_diag = []
    used_proc = []
    med_true = []
    used_med_true = []

    # 分别拼接成一个batch的数据
    for id, visit in enumerate(batch):
        diagnoses_tensor[id, :len(visit[0])] = torch.tensor(visit[0])
        # diagnoses_tensor[id, :len(visit[0])] 表示visit id的diagnose信息，剩下没有diagnose就是-1。当然也可能达到最大的diagnose数
        # tensor([ 93, 228,  29,  13,  33, 467,  15,  18, 439, 103,  35,  21,  37,  23,
        #  76,  51,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
        #  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1])
        # diagnoses_tensor[0] 如上 
        
        procedure_tensor[id, :len(visit[1])] = torch.tensor(visit[1])
        # tensor([59, 60, 48, 62, 96,  2, 82, 46, 61, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # procedure_tensor[0]如上

        medication_tensor[id, :len(visit[2])] = torch.tensor(visit[2])
        # tensor([23, 41, 29, 32, 79,  7, 47, 51, 31, 10, 27, 17, 43,  5,  8, 12, 13,  1,
        #  4,  0, 21,  6,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        #  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        # medication_tensor[0]如上

        # for i in range(len(visit[3])):
        #     used_med_tensor[id, i, :len(visit[3][i])] = torch.tensor(visit[3][i])
        used_med.append(visit[3])
        used_diag.append(visit[4])
        used_proc.append(visit[5])
        med_true.append(visit[6])
        used_med_true.append(visit[7])
    
    return diagnoses_tensor, procedure_tensor, medication_tensor, used_med, used_diag,used_proc,med_true,used_med_true, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix



def pad_batch_v2_eval(batch):
    batch_size = len(batch)

    # 统计每一个visit疾病、手术、药物的最值
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    
    for data in batch: #data就是一个visit
        d_max_num = max(d_max_num, len(data[0]))
        p_max_num = max(p_max_num, len(data[1]))
        m_max_num = max(m_max_num, len(data[2]))

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, d_max_num), -1e9)
    for i in range(batch_size):
        d_mask_matrix[i, :len(batch[i][0])] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, p_max_num), -1e9)
    for i in range(batch_size):
        p_mask_matrix[i, :len(batch[i][1])] = 0.

    # 生成 m_mask_matrix
    m_mask_matrix = torch.full((batch_size, m_max_num), -1e9)
    for i in range(batch_size):
        m_mask_matrix[i, :len(batch[i][2])] = 0.


    # 分别生成diagnoses、procedure、medication的数据
    diagnoses_tensor = torch.full((batch_size, d_max_num), -1)# torch.Size([256, 37])
    procedure_tensor = torch.full((batch_size, p_max_num), -1)#torch.Size([256, 18])
    medication_tensor = torch.full((batch_size, m_max_num), 0)#torch.Size([256, 53])
    used_med = []
    used_diag = []
    used_proc = []
    med_true = []
    used_med_true = []
    
    # 分别拼接成一个batch的数据
    for id, visit in enumerate(batch):
        diagnoses_tensor[id, :len(visit[0])] = torch.tensor(visit[0])        
        procedure_tensor[id, :len(visit[1])] = torch.tensor(visit[1])
        medication_tensor[id, :len(visit[2])] = torch.tensor(visit[2])
        used_med.append(visit[3])
        used_diag.append(visit[4])
        used_proc.append(visit[5])
        med_true.append(visit[6])
        used_med_true.append(visit[7])
        
    return diagnoses_tensor, procedure_tensor, medication_tensor, used_med, used_diag,used_proc,med_true,used_med_true, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix
    

def pad_num_replace(tensor, src_num, target_num):
    # replace_tensor = torch.full_like(tensor, target_num)
    return torch.where(tensor==src_num, target_num, tensor)
    

