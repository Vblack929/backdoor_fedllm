import torch
import numpy as np
import pickle
from .models import SCPN, ParseNet
from .subword import BPE, read_vocabulary
from torch.autograd import Variable

DEFAULT_TEMPLATES = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( VP ) ( . ) ) ) EOP',
    '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
    '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
    '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
]

class SCPNAttacker:
    def __init__(self, model_paths, device=None):
        """
        Initialize SCPN model for attack.
        
        Args:
            model_paths (dict): Paths to model files and vocabularies.
            device (torch.device, optional): Device to use (CPU/GPU).
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.pp_model = torch.load(model_paths['scpn_model'], map_location=self.device)
        self.parse_model = torch.load(model_paths['parse_model'], map_location=self.device)
        self.pp_vocab, self.rev_pp_vocab = pickle.load(open(model_paths['parse_vocab'], 'rb'))
        self.parse_gen_voc = pickle.load(open(model_paths['ptb_tagset'], 'rb'))
        self.rev_label_voc = {v: k for k, v in self.parse_gen_voc.items()}
        
        # Load BPE encoder
        bpe_codes = open(model_paths['bpe_codes'], 'r', encoding='utf-8')
        bpe_vocab = open(model_paths['vocab'], 'r', encoding='utf-8')
        bpe_vocab = read_vocabulary(bpe_vocab, 50)
        self.bpe = BPE(bpe_codes, '@@', bpe_vocab, None)
        
        # Initialize SCPN and ParseNet models
        pp_args = self.pp_model['config_args']
        self.net = SCPN(
            pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans,
            len(self.pp_vocab), len(self.parse_gen_voc) - 1, pp_args.use_input_parse
        ).to(self.device)
        self.net.load_state_dict(self.pp_model['state_dict'])
        self.net.eval()

        parse_args = self.parse_model['config_args']
        self.parse_net = ParseNet(
            parse_args.d_nt, parse_args.d_hid, len(self.parse_gen_voc)
        ).to(self.device)
        self.parse_net.load_state_dict(self.parse_model['state_dict'])
        self.parse_net.eval()

    def gen_paraphrase(self, sentence, templates=DEFAULT_TEMPLATES):
        """
        Generate paraphrases using SCPN.
        
        Args:
            sentence (str): The input sentence to paraphrase.
            templates (list): List of parse templates for paraphrase generation.

        Returns:
            list: A list of paraphrased sentences.
        """
        template_lens = [len(x.split()) for x in templates]
        np_templates = np.zeros((len(templates), max(template_lens)), dtype='int32')
        for z, template in enumerate(templates):
            np_templates[z, :template_lens[z]] = [self.parse_gen_voc[w] for w in template.split()]
        tp_templates = torch.from_numpy(np_templates).long().to(self.device)
        tp_template_lens = torch.LongTensor(template_lens).to(self.device)

        # Tokenize and encode the sentence
        seg_sent = self.bpe.segment(sentence.lower()).split()
        seg_sent = [self.pp_vocab.get(w, self.pp_vocab['UNK']) for w in seg_sent]
        seg_sent.append(self.pp_vocab['EOS'])
        torch_sent = torch.LongTensor(seg_sent).to(self.device)
        torch_sent_len = torch.LongTensor([len(seg_sent)]).to(self.device)

        # Parse the sentence using a constituency parser (assumed to be integrated)
        parse_tree = sentence.split() + ["EOP"]
        torch_parse = torch.LongTensor([self.parse_gen_voc.get(w, self.parse_gen_voc['UNK']) for w in parse_tree]).to(self.device)
        torch_parse_len = torch.LongTensor([len(parse_tree)]).to(self.device)

        # Generate parses from templates
        beam_dict = self.parse_net.batch_beam_search(
            torch_parse.unsqueeze(0), tp_templates, torch_parse_len, tp_template_lens,
            self.parse_gen_voc['EOP'], beam_size=3, max_steps=150
        )

        seq_lens = []
        seqs = []
        for b_idx in beam_dict:
            prob, _, _, seq = beam_dict[b_idx][0]
            seq = seq[:-1]
            seq_lens.append(len(seq))
            seqs.append(seq)

        np_parses = np.zeros((len(seqs), max(seq_lens)), dtype='int32')
        for z, seq in enumerate(seqs):
            np_parses[z, :seq_lens[z]] = seq

        tp_parses = torch.from_numpy(np_parses).long().to(self.device)
        tp_len = torch.LongTensor(seq_lens).to(self.device)

        # Generate paraphrases
        ret = []
        beam_dict = self.net.batch_beam_search(
            torch_sent.unsqueeze(0), tp_parses, torch_sent_len, tp_len,
            self.pp_vocab['EOS'], beam_size=3, max_steps=40
        )

        for b_idx in beam_dict:
            prob, _, _, seq = beam_dict[b_idx][0]
            gen_sent = ' '.join([self.rev_pp_vocab.get(w, '<UNK>') for w in seq[:-1]])
            ret.append(self.reverse_bpe(gen_sent.split()))

        return ret

    def reverse_bpe(self, sent):
        """Reverse the BPE segmentation."""
        x = []
        cache = ''
        for w in sent:
            if w.endswith('@@'):
                cache += w.replace('@@', '')
            elif cache != '':
                x.append(cache + w)
                cache = ''
            else:
                x.append(w)
        return ' '.join(x)