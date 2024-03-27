import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import const

def logsumexp_(input, keepdim=False):
    """adjust the input to avoid overflow and underflow

    Args:
        input (torch.Tensor): two dimension tensor, [bsz, label_size]
        keepdim (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: [bsz, 1] or [bsz]
    """
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)
    output = input - max_scores
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))


def gather_index(input, index):
    assert input.dim() == 2 and index.dim() == 1
    index = index.unsqueeze(1)
    output = torch.gather(input, 1, index)
    
    return output.squeeze(1)


class CRF(nn.Module):
    def __init__(self, label_size):
        super().__init__()
        self.label_size = label_size
        self.transitions = nn.Parameter(
            torch.randn(label_size, label_size))
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.transitions)
        self.transitions.data[const.START, :].fill_(-10000.)
        self.transitions.data[:, const.STOP].fill_(-10000.)

    # 
    def _score_sentence(self, input, tags):
        """Calculate the score of the batched sentence with tags

        Args:
            input (torch.Tensor): batched emission score (output of BiLSTM)
            tags (torch.Tensor): transformed tags

        Returns:
            torch.Tensor: score of the batched sentence
        """
        # tags: [bsz, sent_len]
        bsz, sent_len, label_size = input.size()
        score = torch.FloatTensor(bsz).fill_(0.)
        s_score = torch.LongTensor([[const.START]] * bsz)
        # add START tag 7
        tags = torch.cat([s_score, tags], dim=-1) # [bsz, sent_len + 1]
        input_t = input.transpose(0, 1) # [sent_len, bsz, label_size]

        # add emission score and transition score for each tag
        for i, words in enumerate(input_t):
            # words: [bsz, label_size]
            # tags[:, i]: [bsz] the i-th tag in each sentence
            # each column of transitions is the score from the i-th tag to the next tag
            temp = self.transitions.index_select(1, tags[:, i]) # [label_size, bsz]
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = gather_index(words, tags[:, i + 1])
            score += bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])
        bsz_t = gather_index(temp.transpose(0, 1), torch.LongTensor([const.STOP] * bsz))
        
        return score + bsz_t

    def forward(self, input):
        """calculate the logsumexp of the score of all possible paths for each sentence

        Args:
            input (torch.Tensor): output of the BiLSTM

        Returns:
            torch.Tensor: total score of each sentence, [bsz, 1]
        """
        bsz, sent_len, l_size = input.size()
        init_alphas = torch.FloatTensor(
            bsz, self.label_size).fill_(-10000.)
        init_alphas[:, const.START].fill_(0.)
        forward_var = init_alphas # [bsz, label_size]

        input_t = input.transpose(0, 1) # [sent_len, bsz, label_size]

        # use the forward algorithm to compute the partition function
        for words in input_t: # words: [bsz, label_size]
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].view(-1, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score # [bsz, label_size]
                alphas_t.append(logsumexp_(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1) # [bsz, label_size]
        forward_var += self.transitions[const.STOP].view( 1, -1)
        # return the final score of each sentence
        return logsumexp_(forward_var, True) # [bsz, 1]

    def viterbi_decode(self, input):
        """use the viterbi algorithm to compute the best path for each sentence
        
        Why does it calculate the best path before adding the emission score 
        in each step? Because it add a STOP tag at the end of each sentence, 
        there is no emission score for the STOP word (tag). So what we compute
        at each step is the best path from the current tag to the 
        next tag (not included).
        
        Args:
            input (torch.Tensor): output of the BiLSTM, [sent_len, bsz, label_size]

        Returns:
            torch.Tensor: the best path for each sentence, [bsz, sent_len]
        """
        backpointers = []
        bsz, sent_len, l_size = input.size()

        init_vvars = torch.FloatTensor(bsz, self.label_size).fill_(-10000.)
        init_vvars[:, const.START].fill_(0.)
        forward_var = init_vvars

        input_t = input.transpose(0, 1)

        # viterbi algorithm: compute the best path at each step
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                # the best score from the current tag to the next tag (not included)
                best_tag_scores, best_tag_ids = torch.max(
                    next_tag_var, 1, keepdim=True)  # bsz
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[const.STOP].view(1, -1)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids.view(-1, 1)]
        # back-tracking
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))

        # remove the START tag
        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1) # [bsz, sent_len]


class BiLSTM(nn.Module):
    def __init__(self, word_size, word_ebd_dim, lstm_hsz, lstm_layers, dropout, batch_size):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hsz = lstm_hsz
        self.batch_size = batch_size

        self.word_ebd = nn.Embedding(word_size, word_ebd_dim)
        self.lstm = nn.LSTM(word_ebd_dim,
                            hidden_size=lstm_hsz // 2,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        
        self._init_weights()


    def _init_weights(self):
        torch.nn.init.uniform_(self.word_ebd.weight, -1, 1)

    def forward(self, words, seq_lengths):
        encode = self.word_ebd(words)
        # pad the sequence to the longest length
        packed_encode = torch.nn.utils.rnn.pack_padded_sequence(encode, seq_lengths, batch_first=True)
        packed_output, _ = self.lstm(packed_encode)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        return output 


class Model(nn.Module):
    def __init__(self, word_size, word_ebd_dim, lstm_hsz, 
                 lstm_layers, label_size, dropout, batch_size):
        super().__init__()

        self.word_size = word_size
        self.word_ebd_dim = word_ebd_dim
        self.lstm_hsz = lstm_hsz
        self.lstm_layers = lstm_layers
        self.label_size = label_size
        self.dropout = dropout
        self.batch_size = batch_size
        

        self.bilstm = BiLSTM(self.word_size, self.word_ebd_dim,
                             self.lstm_hsz, self.lstm_layers, self.dropout, self.batch_size)

        self.fc1 = nn.Linear(self.lstm_hsz, self.label_size)
        self.crf = CRF(self.label_size)
        self._init_weights()

    def forward(self, words, labels, seq_lengths):
        output = self.bilstm(words, seq_lengths)
        output = self.fc1(output)
        pre_score = self.crf(output)
        label_score = self.crf._score_sentence(output, labels)
        return (pre_score - label_score).mean()

    def predict(self, word, seq_lengths):
        lstm_out = self.bilstm(word, seq_lengths)
        out = self.fc1(lstm_out)
        return self.crf.viterbi_decode(out)

    def _init_weights(self, scope=1.):
        self.fc1.weight.data.uniform_(-scope, scope)
        self.fc1.bias.data.fill_(0)
