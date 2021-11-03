# 2021 Dongji Gao

import os
from pathlib import Path


class Aligner():
    def __init__(self, model, dataset, lang_dir, graph_dir):
        self.model = model
        self.dataset = dataset
        self.lang_dir = Path(lang_dir)
        self.graph_dir = Path(graph_dir)

    def make_acceptor(self):
        raise NotImplementedError

    def make_graph(self):
        raise NotImplementedError

    def load_graph(self):
        pass

    def decode(self):
        pass


class FlexibleAligner(Aligner):
    def __init__(self, model, dataset, graph_dir):
        super().__init__(model, dataset, graph_dir)
        pass

    def make_accptor(self, graph_dir, weight=0, skip_weight=0, allow_deletion=True):
        with open(graph_dir / "G.fst.txt", w) as G:
            for text in texts:
                start_state = 0
                final_state = 1
                cur_state = start_state
                next_state = 2

                line_list = text.split()
                assert (len(line_list) > 1)

                for word in line_list[1:-1]:
                    G.write(f"{cur_state}\t{next_state}\t{word}\t{word}\t{weight}\n")
                    if args.allow_deletion:
                        G.write(f"{start_state}\t{next_state}\t{word}\t{word}\t{skip_weight}\n")
                        G.write(f"{next_state}\t{final_state}\t{word}\t{word}\t{skip_weight}\n")
                    cur_state = next_state
                    next_state += 1

                # final state
                word = line_list[-1]
                G.write(f"{cur_state}\t{final_state}\t{word}\t{word}\t{weight}\n")
                G.write(f"{final_state}\t0\n\n")

        def make_graph(self, lang_dir, graph_dir):
            G_list = list()

            phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
            word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
            phone_ids = get_phone_symbols(phone_symbol_table)
            phone_ids_with_blank = [0] + phone_ids
            ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

            with open(graph_dir / "L_disambig.fst.txt") as f:
                L = k2.Fsa.from_openfst(f.read(), acceptor=False)
                print("L loaded")
            with open(graph_dir / "G.fst.txt") as f:
                if task == "alignment":
                    G_all = f.read().strip().split("\n\n")
                    for G_single in G_all:
                        G = k2.Fsa.from_openfst(G_single, acceptor=False)
                        G_list.append(G)
                    G = k2.create_fsa_vec(G_list)
                else:
                    G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            print("G loaded")

            first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
            first_word_disambig_id = find_first_disambig_symbol(word_symbol_table)

            HLG = compile_HLG(L=L,
                              G=G,
                              H=ctc_topo,
                              labels_disambig_id_start=first_phone_disambig_id,
                              aux_labels_disambig_id_start=first_word_disambig_id)

            torch.save(HLG.as_dict(), graph_dir / "HLG.pt")
