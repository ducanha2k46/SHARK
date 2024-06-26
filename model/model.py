import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel

class GuidedMoEBasic(nn.Module):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=2, guiding_lambda=0, **kwargs):
        super(GuidedMoEBasic, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.emotion_linear = nn.Linear(self.bert.config.hidden_size, n_emotion)
        self.n_expert = n_expert
        self.guiding_lambda = guiding_lambda
        self.gating_network = nn.Linear(2 * (self.bert.config.hidden_size + n_emotion + 1), n_expert)
        self.cause_linear = nn.ModuleList()

        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.bert.config.hidden_size + n_emotion + 1), 256),
                                                    nn.Linear(256, n_cause)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, emotion_pred, h_prime, input_ids, speaker_ids):
        cause_pred = self.binary_cause_classification_task(emotion_pred, input_ids, h_prime, speaker_ids)
        return cause_pred


    def binary_cause_classification_task(self, emotion_prediction, input_ids, h_prime, speaker_ids):
        pair_embedding = self.get_pair_embedding(emotion_prediction, input_ids, h_prime, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())
        print("gating_prob.size()")
        print(gating_prob.size())
        gating_prob = self.guiding_lambda * self.get_subtask_label(input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1, self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return cause_pred

    def gating_network_train(self, emotion_prediction, input_ids, h_prime, speaker_ids):
        pair_embedding = self.get_pair_embedding(emotion_prediction, input_ids, h_prime, speaker_ids)
        return self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

    def get_pair_embedding(self, emotion_prediction, input_ids, h_prime, speaker_ids):
        max_doc_len, x = input_ids.size()
        batch_size = 1
        print(input_ids.size())
        print(emotion_prediction.size())
        print(h_prime.size())
        
       
        emotion_prediction = torch.mean(emotion_prediction, dim = 1)
        h_prime =  torch.mean(h_prime, dim = 1)
        print(input_ids.size())
        print(emotion_prediction.size())
        print(h_prime.size())
        print(speaker_ids.view(-1).unsqueeze(1).size())
       
        concatenated_embedding = torch.cat((h_prime, emotion_prediction, speaker_ids.unsqueeze(1)), dim=1) # 여기서 emotion_prediction에 detach를 해야 문제가 안생기겠지? 해보고 문제생기면 detach 고고

        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1):
            pair_per_batch = list()
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t]))) # backward 시, cycle이 생겨 문제가 생길 경우, batch[end_t].detach() 시도.
            pair_embedding.append(torch.stack(pair_per_batch))
        
        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)
        print("pair_embedding.size()")
        print(pair_embedding.size())
        return pair_embedding

    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        # After Inheritance, Define function.
        pass

class PRG_MoE(GuidedMoEBasic):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=4, guiding_lambda=0, **kwargs):
        super().__init__(dropout=dropout, n_speaker=n_speaker, n_emotion=n_emotion, n_cause=n_cause, n_expert=4, guiding_lambda=guiding_lambda)

    # def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
    #     most_common_numbers = []
    #     probabilities = torch.argmax(emotion_prediction, dim=-1).cpu().numpy()
    #     for i, sentence in enumerate(probabilities):
    #         num_characters = speaker_ids[i]
            
    #         counts = np.bincount(sentence[:num_characters])
            
    #         sorted_counts = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
            
    #         most_common = 0
    #         if sorted_counts[0][0] == 0:
    #             if len(sorted_counts) > 1:
    #                 most_common = sorted_counts[1][0]
    #         else:
    #             most_common = sorted_counts[0][0]
            
    #         most_common_numbers.append(most_common)
        
    #     print(most_common_numbers)
        
    #     pair_info = []
    #     for i in range(len(input_ids)):
    #         for j in range(i + 1, len(input_ids)):
    #             if (i % 2 == j % 2):  
    #                 speaker_same = True
    #             else:
    #                 speaker_same = False
    #             emotion_same = most_common_numbers[i] == most_common_numbers[j]  
    #             if speaker_same and emotion_same:
    #                 pair_info.append(torch.tensor([1, 0, 0, 0]))
    #             elif speaker_same and not emotion_same:
    #                 pair_info.append(torch.tensor([0, 1, 0, 0]))
    #             elif not speaker_same and emotion_same:
    #                 pair_info.append(torch.tensor([0, 0, 1, 0]))
    #             else:
    #                 pair_info.append(torch.tensor([0, 0, 0, 1]))

    #     pair_info = torch.stack(pair_info).to(input_ids.device)
    #     print("pair_info:")
    #     print(pair_info.size())
    #     return pair_info

    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        batch_size = 1
        max_doc_len, x= input_ids.size()
        most_common_numbers = []
        probabilities = torch.argmax(emotion_prediction, dim=-1).cpu().numpy()
        for i, sentence in enumerate(probabilities):
            num_characters = speaker_ids[i]
            
            counts = np.bincount(sentence[:num_characters])
            
            sorted_counts = sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
            
            most_common = 0
            if sorted_counts[0][0] == 0:
                if len(sorted_counts) > 1:
                    most_common = sorted_counts[1][0]
            else:
                most_common = sorted_counts[0][0]
            
            most_common_numbers.append(most_common)
        
        pair_info = []
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = []
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    if (end_t % 2 == t % 2):  
                        speaker_condition = True
                    else:
                        speaker_condition = False
                    emotion_condition = most_common_numbers[t] == most_common_numbers[end_t]

                    if speaker_condition and emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0])) # if speaker and dominant emotion are same
                    elif speaker_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0])) # if speaker is same, but dominant emotion is differnt
                    elif emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0])) # if speaker is differnt, but dominant emotion is same
                    else:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1])) # if speaker and dominant emotion are differnt
            pair_info.append(torch.stack(info_pair_per_batch))
        
        pair_info = torch.stack(pair_info).to(input_ids.device)
        print("pair_info")
        print(pair_info.size())
        return pair_info