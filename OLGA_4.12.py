"""
Author: chensy, Zhangzy，Zhangrh, Licz,Liucx
"""
import cProfile
import re
import argparse
import numpy as np
from preprocess_generative_model_and_data import PreprocessedParametersVDJ, PreprocessedParametersVJ
from utils import nt2codon_rep, gene_to_num_str
import olga.load_model as load_model

# //////////////////////来自于父类GenerationProbability的函数
class GenerationProbability(object):
    """Class used to define Pgen functions and sequence formatting.

    This class is used to define three types of functions that are used by
    both the VDJ pgen algorithm and the VJ pgen algorithm.

    The first type is functions that wrap around the core 'amino acid'
    algorithms to allow for computing Pgen of regular expression, amino acid,
    and nucleotide CDR3 sequences (etc).

    The second type are functions that format some of the inputs (V/J mask,
    lists seqs for regular expressions) of the first type.

    The last group of functions are alignment/matching scripts that are used to
    check how much of an 'amino acid' CDR3 is consistent with a given
    nucleotide sequence. These methods are used in both core algorithms when
    including the V and J contributions.

    Attributes
    ----------
    codons_dict : dict
        Dictionary, keyed by the allowed 'amino acid' symbols with the values
        being lists of codons corresponding to the symbol.

    d_V_usage_mask : list of int
        Default V usage mask of indices of all productive V genes/alleles.
    V_mask_mapping : dict
        Dictionary mapping allowed keywords (strings corresponding to V gene
        and V allele names) to the indices of the V alleles they refer to.

    d_J_usage_mask : list of int
        Default J usage mask of indices of all productive J genes/alleles.
    J_mask_mapping : dict
        Dictionary mapping allowed keywords (strings corresponding to J gene
        and J allele names) to the indices of the J alleles they refer to.

    Methods
    ----------
    compute_regex_CDR3_template_pgen(regex_seq, V_usage_mask_in = None, J_usage_mask_in = None, print_warnings = True, raise_overload_warning = True)
        Compute Pgen for all seqs consistent with regular expression regex_seq.

    compute_aa_CDR3_pgen(CDR3_seq, V_usage_mask_in = None, J_usage_mask_in = None, print_warnings = True)
        Compute Pgen for the amino acid sequence CDR3_seq

    compute_hamming_dist_1_pgen(CDR3_seq, V_usage_mask_in = None, J_usage_mask_in = None, print_warnings = True)
        Compute Pgen of all seqs hamming dist 1 (in amino acids) from CDR3_seq

    compute_nt_CDR3_pgen(CDR3_ntseq, V_usage_mask_in = None, J_usage_mask_in = None, print_warnings = True)
        Compute Pgen for the inframe nucleotide sequence CDR3_ntseq.

    compute_CDR3_pgen(CDR3_seq, V_usage_mask, J_usage_mask)
        Dummy function that is replaced in classes GenerationProbabilityV(D)J.
        The methods that replace it implement the different algorithms for
        computing Pgen on a VDJ CDR3 sequence or a VJ CDR3 sequence.

    format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings = True)
        Format raw usage masks into lists of indices.

    list_seqs_from_regex(regex_seq, print_warnings = True, raise_overload_warning = True)
        List sequences that match regular expression template.

    max_nt_to_aa_alignment_left(CDR3_seq, ntseq)
        Find maximum match between CDR3_seq and ntseq from the left.

    max_nt_to_aa_alignment_right(CDR3_seq, ntseq)
        Find maximum match between CDR3_seq and ntseq from the right.

    """

    def __init__(self):
        """Initialize class GenerationProbability.

        Only define dummy attributes for this class. The children classes
        GenerationProbabilityVDJ and GenerationProbabilityVJ will initialize
        the actual attributes.

        """

        self.codons_dict = None
        self.d_V_usage_mask = None
        self.V_mask_mapping = None

        self.d_J_usage_mask = None
        self.J_mask_mapping = None

    def compute_regex_CDR3_template_pgen(self, regex_seq, V_usage_mask_in=None, J_usage_mask_in=None,
                                         print_warnings=True, raise_overload_warning=True):
        """Compute Pgen for all seqs consistent with regular expression regex_seq.

        Computes Pgen for a (limited vocabulary) regular expression of CDR3
        amino acid sequences, conditioned on the V genes/alleles indicated in
        V_usage_mask_in and the J genes/alleles in J_usage_mask_in. Please note
        that this function will list out all the sequences that correspond to the
        regular expression and then calculate the Pgen of each sequence in
        succession. THIS CAN BE SLOW. Consider defining a custom alphabet to
        represent any undetermined amino acids as this will greatly speed up the
        computations. For example, if the symbol ^ is defined as [AGR] in a custom
        alphabet, then instead of running
        compute_regex_CDR3_template_pgen('CASS[AGR]SARPEQFF', ppp),
        which will compute Pgen for 3 sequences, the single sequence
        'CASS^SARPEQFF' can be considered. (Examples are TCRB sequences/model)


        Parameters
        ----------
        regex_seq : str
            The regular expression string that represents the CDR3 sequences to be
            listed then their Pgens computed and summed.
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.
        raise_overload_warning : bool
            A flag to warn of more than 10000 seqs corresponding to the regex_seq

        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence

        Examples
        --------
        >>> generation_probability.compute_regex_CDR3_template_pgen('CASS[AGR]SARPEQFF')
        8.1090898050318022e-10
        >>> generation_probability.compute_regex_CDR3_template_pgen('CASSAX{0,5}SARPEQFF')
        6.8468778040965569e-10

        """

        V_usage_mask, J_usage_mask = self.format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings)

        CDR3_seqs = self.list_seqs_from_regex(regex_seq, print_warnings, raise_overload_warning)

        pgen = 0
        for CDR3_seq in CDR3_seqs:
            if len(CDR3_seq) == 0:
                continue
            pgen += self.compute_CDR3_pgen(CDR3_seq, V_usage_mask, J_usage_mask)

        return pgen

    def compute_aa_CDR3_pgen(self, CDR3_seq, V_usage_mask_in=None, J_usage_mask_in=None, print_warnings=True):
        """Compute Pgen for the amino acid sequence CDR3_seq.

        Conditioned on the V genes/alleles indicated in V_usage_mask_in and the
        J genes/alleles in J_usage_mask_in. (Examples are TCRB sequences/model)

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' -- the standard amino acids,
            plus any custom symbols for an expanded codon alphabet (note the
            standard ambiguous amino acids -- B, J, X, and Z -- are included by
            default).
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.

        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence

        Examples
        --------
        >>> generation_probability.compute_aa_CDR3_pgen('CAWSVAPDRGGYTF')
        1.5756106696284584e-10
        >>> generation_probability.compute_aa_CDR3_pgen('CAWSVAPDRGGYTF', 'TRBV30*01', 'TRBJ1-2*01')
        1.203646865765782e-10
        >>> generation_probability.compute_aa_CDR3_pgen('CAWXXXXXXXGYTF')
        7.8102586432014974e-05

        """
        if len(CDR3_seq) == 0:
            return 0
        for aa in CDR3_seq:
            if aa not in self.codons_dict.keys():
                # Check to make sure all symbols are accounted for
                if print_warnings:
                    print('Invalid amino acid CDR3 sequence --- unfamiliar symbol: ' + aa)
                return 0

        V_usage_mask, J_usage_mask = self.format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings)

        return self.compute_CDR3_pgen(CDR3_seq, V_usage_mask, J_usage_mask)

    def compute_hamming_dist_1_pgen(self, CDR3_seq, V_usage_mask_in=None, J_usage_mask_in=None, print_warnings=True):
        """Compute Pgen of all seqs hamming dist 1 (in amino acids) from CDR3_seq.

        Please note that this function will list out all the
        sequences that are hamming distance 1 from the base sequence and then
        calculate the Pgen of each sequence in succession. THIS CAN BE SLOW
        as it computes Pgen for L+1 sequences where L = len(CDR3_seq). (Examples
        are TCRB sequences/model)

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of amino acids (ONLY the standard amino acids).
            Pgens for all sequences of hamming distance 1 (in amino acid sequence)
            are summed.
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.

        Returns
        -------
        pgen : float
            The sum of generation probabilities (Pgens) of the sequences at most
            hamming distance 1 (in amino acids) from CDR3_seq.

        """

        # make sure that the symbol X is defined as the fully undetermined amino acid:
        # X ~ ACDEFGHIKLMNPQRSTVWY

        V_usage_mask, J_usage_mask = self.format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings)

        if len(CDR3_seq) == 0:
            return 0
        for aa in CDR3_seq:
            if aa not in 'ACDEFGHIKLMNPQRSTVWY':
                # Check to make sure all symbols are accounted for
                if print_warnings:
                    print('Invalid amino acid CDR3 sequence --- unfamiliar symbol: ' + aa)
                return 0
        tot_pgen = 0
        for i in range(len(CDR3_seq)):
            tot_pgen += self.compute_CDR3_pgen(CDR3_seq[:i] + 'X' + CDR3_seq[i + 1:], V_usage_mask, J_usage_mask)
        tot_pgen += -(len(CDR3_seq) - 1) * self.compute_CDR3_pgen(CDR3_seq, V_usage_mask, J_usage_mask)
        return tot_pgen

    def compute_nt_CDR3_pgen(self, CDR3_ntseq, V_usage_mask_in=None, J_usage_mask_in=None, print_warnings=True):
        """Compute Pgen for the inframe nucleotide sequence CDR3_ntseq.

        Conditioned on the V genes/alleles indicated in V_usage_mask_in and the
        J genes/alleles in J_usage_mask_in. (Examples are TCRB sequences/model)

        Parameters
        ----------
        CDR3_ntseq : str
            Inframe nucleotide sequence composed of ONLY A, C, G, or T (either
            uppercase or lowercase).
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.

        Returns
        -------
        pgen : float64
            The generation probability (Pgen) of the sequence

        Examples
        --------
        >>> generation_probability.compute_nt_CDR3_pgen('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC')
        3.2674893012379071e-12
        >>> generation_probability.compute_nt_CDR3_pgen('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC', 'TRBV30*01', 'TRBJ1-2*01')
        2.3986503758867323e-12

        """

        if not len(CDR3_ntseq) % 3 == 0:
            # Make sure sequence is inframe
            if print_warnings:
                print('Invalid nucleotide CDR3 sequence --- out of frame sequence')
            return 0
        elif len(CDR3_ntseq) == 0:
            return 0
        else:
            for nt in CDR3_ntseq:
                if nt not in 'ACGTacgt':
                    if print_warnings:
                        print('Invalid nucleotide CDR3 sequence --- unfamiliar nucleotide: ' + nt)
                    return 0

        V_usage_mask, J_usage_mask = self.format_usage_masks(V_usage_mask_in, J_usage_mask_in, print_warnings)

        return self.compute_CDR3_pgen(nt2codon_rep(CDR3_ntseq), V_usage_mask, J_usage_mask)

    def compute_CDR3_pgen(self, CDR3_seq, V_usage_mask, J_usage_mask):
        """Dummy function that is replaced in classes GenerationProbabilityV(D)J."""
        # Proxy for the actual function that will call either the VDJ algorithm
        # or the VJ algorithm
        pass

    # Formatting methods for the top level Pgen computation calls
    def format_usage_masks(self, V_usage_mask_in, J_usage_mask_in, print_warnings=True):
        """Format raw usage masks into lists of indices.

        Usage masks allows the Pgen computation to be conditioned on the V and J
        gene/allele identities. The inputted masks are lists of strings, or a
        single string, of the names of the genes or alleles to be conditioned on.
        The default mask includes all productive V or J genes.

        Parameters
        ----------
        V_usage_mask_in : str or list
            An object to indicate which V alleles should be considered. The default
            input is None which returns the list of all productive V alleles.
        J_usage_mask_in : str or list
            An object to indicate which J alleles should be considered. The default
            input is None which returns the list of all productive J alleles.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.

        Returns
        -------
        V_usage_mask : list of integers
            Indices of the V alleles to be considered in the Pgen computation
        J_usage_mask : list of integers
            Indices of the J alleles to be considered in the Pgen computation

        Examples
        --------
        >>> generation_probability.format_usage_masks('TRBV27*01','TRBJ1-1*01')
        ([34], [0])
        >>> generation_probability.format_usage_masks('TRBV27*01', '')
        ([34], [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13])
        >>> generation_probability.format_usage_masks(['TRBV27*01', 'TRBV13*01'], 'TRBJ1-1*01')
        ([34, 18], [0])

        """
        # Format the V usage mask
        if isinstance(V_usage_mask_in, str):
            V_usage_mask_in = [V_usage_mask_in]

        if V_usage_mask_in is None:  # Default case, use all productive V genes with non-zero probability
            # V_usage_mask = [v for v, V in enumerate(ppp['cutV_genomic_CDR3_segs']) if len(V) > 0]
            V_usage_mask = self.d_V_usage_mask
        elif isinstance(V_usage_mask_in, list):
            e_V_usage_mask = set()
            for v in V_usage_mask_in:
                try:
                    e_V_usage_mask = e_V_usage_mask.union(self.V_mask_mapping[gene_to_num_str(v, 'V')])
                except:
                    if print_warnings:
                        print('Unfamiliar V gene/allele: ' + v)
                    pass
            if len(e_V_usage_mask) == 0:
                if print_warnings:
                    print('No recognized V genes/alleles. Using default V_usage_mask')
                V_usage_mask = self.d_V_usage_mask
            else:
                V_usage_mask = list(e_V_usage_mask)
        else:
            if print_warnings:
                print('Unfamiliar typed V usage mask: ' + str(V_usage_mask_in) + '. Using default V_usage_mask')
            V_usage_mask = self.d_V_usage_mask

        # Format the J usage mask
        if isinstance(J_usage_mask_in, str):
            J_usage_mask_in = [J_usage_mask_in]

        if J_usage_mask_in is None:  # Default case, use all productive J genes with non-zero probability
            # J_usage_mask = [j for j, J in enumerate(ppp['cutJ_genomic_CDR3_segs']) if len(J) > 0]
            J_usage_mask = self.d_J_usage_mask
        elif isinstance(J_usage_mask_in, list):
            e_J_usage_mask = set()
            for j in J_usage_mask_in:
                try:
                    e_J_usage_mask = e_J_usage_mask.union(self.J_mask_mapping[gene_to_num_str(j, 'J')])
                except:
                    if print_warnings:
                        print('Unfamiliar J gene/allele: ' + j)
                    pass
            if len(e_J_usage_mask) == 0:
                if print_warnings:
                    print('No recognized J genes/alleles. Using default J_usage_mask')
                J_usage_mask = self.d_J_usage_mask
            else:
                J_usage_mask = list(e_J_usage_mask)
        else:
            if print_warnings:
                print('Unfamiliar typed J usage mask: ' + str(J_usage_mask_in) + '. Using default J_usage_mask')
            J_usage_mask = self.d_J_usage_mask

        return V_usage_mask, J_usage_mask

    def list_seqs_from_regex(self, regex_seq, print_warnings=True, raise_overload_warning=True):
        """List sequences that match regular expression template.

        This function parses a limited regular expression vocabulary, and
        lists all the sequences consistent with the regular expression. Supported
        regex syntax: [] and {}. Cannot have two {} in a row. Note we can't use
        Kline star (*) as this is the symbol for a stop codon --- use {}.

        Parameters
        ----------
        regex_seq : str
            The regular expression string that represents the sequences to be
            listed.
        print_warnings : bool
            Determines whether warnings are printed or not. Default ON.
        raise_overload_warning : bool
            A flag to warn of more than 10000 seqs corresponding to the regex_seq

        Returns
        -------
        CDR3_seqs : list
            A list of CDR3 sequences that correspond to the regex_seq

        Examples
        --------
        >>> generation_probability.list_seqs_from_regex('CASS[AGR]SARPEQFF')
        ['CASSGSARPEQFF', 'CASSRSARPEQFF', 'CASSASARPEQFF']
        >>> generation_probability.list_seqs_from_regex('CASSAX{0,5}SARPEQFF')
        ['CASSASARPEQFF',
         'CASSAXXXXSARPEQFF',
         'CASSAXXSARPEQFF',
         'CASSAXXXXXSARPEQFF',
         'CASSAXXXSARPEQFF',
         'CASSAXSARPEQFF']

        """

        aa_symbols = ''.join(self.codons_dict)

        default_max_reps = 40

        # Check to make sure that expression is of the right form/symbols

        # Identify bracket expressions
        bracket_ex = [x for x in re.findall('\[[' + aa_symbols + ']*?\]|\{\d+,{0,1}\d*\}', regex_seq)]

        split_seq = re.split('\[[' + aa_symbols + ']*?\]|\{\d+,{0,1}\d*\}', regex_seq)
        # Check that all remaining characters are in the codon dict
        for aa in ''.join(split_seq):
            if aa not in aa_symbols:
                if print_warnings:
                    print(
                        'Unfamiliar symbol representing a codon:' + aa + ' --- check codon dictionary or the regex syntax')
                return []

        regex_list = [split_seq[i // 2] if i % 2 == 0 else bracket_ex[i // 2] for i in
                      range(len(bracket_ex) + len(split_seq)) if not (i % 2 == 0 and len(split_seq[i // 2]) == 0)]

        max_num_seqs = 1
        for l, ex in enumerate(regex_list[::-1]):
            i = len(regex_list) - l - 1
            if ex[0] == '[':  # bracket expression
                # check characters
                for aa in ex.strip('[]'):
                    if aa not in aa_symbols:
                        if print_warnings:
                            print('Unfamiliar symbol representing a codon:' + aa + ' --- check codon dictionary')
                        return []
                max_num_seqs *= len(ex) - 2
            elif ex[0] == '{':  # curly bracket
                if i == 0:
                    if print_warnings:
                        print("Can't have {} expression at start of sequence")
                    return []
                elif isinstance(regex_list[i - 1], list):
                    if print_warnings:
                        print("Two {} expressions in a row is not supported")
                    return []
                elif regex_list[i - 1][0] == '[':
                    syms = regex_list[i - 1].strip('[]')
                    regex_list[i - 1] = ''
                else:
                    syms = regex_list[i - 1][-1]
                    regex_list[i - 1] = regex_list[i - 1][:-1]
                if ',' not in ex:
                    new_expression = [int(ex.strip('{}')), int(ex.strip('{}')), syms]
                    max_num_seqs *= len(syms) ** new_expression[0]
                else:
                    try:
                        new_expression = [int(ex.strip('{}').split(',')[0]), int(ex.strip('{}').split(',')[1]), syms]
                    except ValueError:  # No max limit --- use default
                        new_expression = [int(ex.strip('{}').split(',')[0]), default_max_reps, syms]
                    if new_expression[0] > new_expression[1]:
                        if print_warnings:
                            print('Check regex syntax --- should be {min,max}')
                        return []
                    max_num_seqs *= sum(
                        [len(syms) ** n for n in range(new_expression[0], new_expression[1] + 1)]) / len(syms)
                # print new_expression
                regex_list[i] = new_expression

        if max_num_seqs > 10000 and raise_overload_warning:
            if print_warnings:
                answer = input('Warning large number of sequences (estimated ' + str(
                    max_num_seqs) + ' seqs) match the regular expression. Possible memory and time issues. Continue? (y/n)')
                if not answer == 'y':
                    print('Canceling...')
                    return []
            else:
                return []
        # print regex_list
        CDR3_seqs = ['']
        for l, ex in enumerate(regex_list[::-1]):
            i = len(regex_list) - l - 1
            if isinstance(ex, list):  # curly bracket case
                c_seqs = ['']
                f_seqs = []
                for j in range(ex[1] + 1):
                    if j in range(ex[0], ex[1] + 1):
                        f_seqs += c_seqs
                    c_seqs = [aa + c_seq for aa in ex[2] for c_seq in c_seqs]
                CDR3_seqs = [f_seq + CDR3_seq for f_seq in f_seqs for CDR3_seq in CDR3_seqs]
            elif len(ex) == 0:
                pass
            elif ex[0] == '[':  # square bracket case
                CDR3_seqs = [aa + CDR3_seq for aa in ex.strip('[]') for CDR3_seq in CDR3_seqs]
            else:
                CDR3_seqs = [ex + CDR3_seq for CDR3_seq in CDR3_seqs]

        return list(set(CDR3_seqs))

    # Alignment/Matching methods
    def max_nt_to_aa_alignment_left(self, CDR3_seq, ntseq):
        """Find maximum match between CDR3_seq and ntseq from the left.

        This function returns the length of the maximum length nucleotide
        subsequence of ntseq contiguous from the left (or 5' end) that is
        consistent with the 'amino acid' sequence CDR3_seq.

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        ntseq : str
            Genomic (V locus) nucleotide sequence to match.

        Returns
        -------
        max_alignment : int
            Maximum length (in nucleotides) nucleotide sequence that matches the
            CDR3 'amino acid' sequence.

        Example
        --------
        >>> generation_probability.max_nt_to_aa_alignment_left('CASSSEGAGGPSLRGHEQFF', 'TGTGCCAGCAGTTTATCGATA')
        13

        """

        max_alignment = 0
        if len(ntseq) == 0:
            return 0
        aa_aligned = True
        while aa_aligned:
            if ntseq[max_alignment:max_alignment + 3] in self.codons_dict[CDR3_seq[max_alignment // 3]]:
                max_alignment += 3
                if max_alignment // 3 == len(CDR3_seq):
                    return max_alignment
            else:
                break
                aa_aligned = False
        last_codon = ntseq[max_alignment:max_alignment + 3]
        codon_frag = ''
        for nt in last_codon:
            codon_frag += nt
            if codon_frag in self.sub_codons_left[CDR3_seq[max_alignment // 3]]:
                max_alignment += 1
            else:
                break
        return max_alignment

    def max_nt_to_aa_alignment_right(self, CDR3_seq, ntseq):
        """Find maximum match between CDR3_seq and ntseq from the right.

        This function returns the length of the maximum length nucleotide
        subsequence of ntseq contiguous from the right (or 3' end) that is
        consistent with the 'amino acid' sequence CDR3_seq

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        ntseq : str
            Genomic (J locus) nucleotide sequence to match.

        Returns
        -------
        max_alignment : int
            Maximum length (in nucleotides) nucleotide sequence that matches the
            CDR3 'amino acid' sequence.

        Example
        --------
        >>> generation_probability.max_nt_to_aa_alignment_right('CASSSEGAGGPSLRGHEQFF', 'TTCATGAACACTGAAGCTTTCTTT')
        6

        """
        r_CDR3_seq = CDR3_seq[::-1]  # reverse CDR3_seq
        r_ntseq = ntseq[::-1]  # reverse ntseq
        max_alignment = 0
        if len(ntseq) == 0:
            return 0
        aa_aligned = True
        while aa_aligned:
            if r_ntseq[max_alignment:max_alignment + 3][::-1] in self.codons_dict[r_CDR3_seq[max_alignment // 3]]:
                max_alignment += 3
                if max_alignment // 3 == len(CDR3_seq):
                    return max_alignment
            else:
                break
                aa_aligned = False
        r_last_codon = r_ntseq[max_alignment:max_alignment + 3]
        codon_frag = ''
        for nt in r_last_codon:
            codon_frag = nt + codon_frag
            if codon_frag in self.sub_codons_right[r_CDR3_seq[max_alignment // 3]]:
                max_alignment += 1
            else:
                break
        return max_alignment

    def max_nt_to_nt_alignment_left(self, CDR3_seq_nt, ntseq):
        """Find maximum match between CDR3_seq and ntseq from the left.

        This function returns the length of the maximum length nucleotide
        subsequence of ntseq contiguous from the left (or 5' end) that is
        consistent with the 'amino acid' sequence CDR3_seq.

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        ntseq : str
            Genomic (V locus) nucleotide sequence to match.

        Returns
        -------
        max_alignment : int
            Maximum length (in nucleotides) nucleotide sequence that matches the
            CDR3 'amino acid' sequence.

        Example
        --------
        >>> generation_probability.max_nt_to_aa_alignment_left('CASSSEGAGGPSLRGHEQFF', 'TGTGCCAGCAGTTTATCGATA')
        13

        """

        max_alignment = 0
        '''
        if len(ntseq) == 0:
            return 0
        aa_aligned = True
        while aa_aligned:
            if ntseq[max_alignment:max_alignment + 3] in self.codons_dict[CDR3_seq[max_alignment // 3]]:
                max_alignment += 3
                if max_alignment // 3 == len(CDR3_seq):
                    return max_alignment
            else:
                break
                aa_aligned = False
        last_codon = ntseq[max_alignment:max_alignment + 3]
        codon_frag = ''
        for nt in last_codon:
            codon_frag += nt
            if codon_frag in self.sub_codons_left[CDR3_seq[max_alignment // 3]]:
                max_alignment += 1
            else:
                break
        '''
        for pos in range(len(ntseq)):
            if ntseq[pos] == CDR3_seq_nt[pos]:
                max_alignment += 1
            else:
                break
        return max_alignment

    def max_nt_to_nt_alignment_right(self, CDR3_seq_nt, ntseq):
        r_CDR3_seq_nt = CDR3_seq_nt[::-1]  # reverse CDR3_seq
        r_ntseq = ntseq[::-1]  # reverse ntseq
        max_alignment = 0
        for pos in range(len(r_ntseq)):
            if ntseq[pos] == r_CDR3_seq_nt[pos]:
                max_alignment += 1
            else:
                break
        return max_alignment


class GenerationProbabilityVDJ(GenerationProbability, PreprocessedParametersVDJ):

    def __init__(self, generative_model, genomic_data, alphabet_file=None):
        GenerationProbability.__init__(self)
        PreprocessedParametersVDJ.__init__(self, generative_model, genomic_data, alphabet_file)

    def compute_Pi_V_nt(self, CDR3_seq_nt, V_usage_mask):
        Pi_V = np.zeros((1, len(CDR3_seq_nt)))  # Holds the aggregate weight for each nt possiblity and position
        alignment_lengths = []
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        for V_in in V_usage_mask:
            # 下面这个try相当于一个检验函数，对运算本身没有影响
            try:
                cutV_gen_seg = self.cutV_genomic_CDR3_segs[V_in]  ##函数来自于load_model,产生一个标准的修建好的V基因
            except IndexError:
                print('Check provided V usage mask. Contains indicies out of allowed range.')
                continue  # 保证修剪好的基因和mask名能够一一对应

            # section2
            current_alignment_length = self.max_nt_to_nt_alignment_left(CDR3_seq_nt, cutV_gen_seg)
            alignment_lengths += [current_alignment_length]
            current_Pi_V = np.zeros((1, len(CDR3_seq_nt)))  # 存概率的向量

            if current_alignment_length > 0:
                for pos in range(len(cutV_gen_seg)):
                    current_Pi_V[0, pos] = self.PVdelV_nt_pos_vec[V_in][nt2num[cutV_gen_seg[pos]], pos]  # 这个文件已提取

                ##section3
                for pos in range(1, current_alignment_length, 3):  # 以步长为三地，向每个密码子的第二位，向矩阵内部添加东西进去,这个矩阵只会第二位有数据
                    condon = cutV_gen_seg[pos - 1] + cutV_gen_seg[pos] + cutV_gen_seg[pos + 1]
                    current_Pi_V[0, pos] = self.PVdelV_2nd_nt_pos_per_aa_vec[V_in][self.codons_dict[condon]][
                        nt2num[cutV_gen_seg[pos]], pos]  # 这个文件已提取
                Pi_V[0, :current_alignment_length] += current_Pi_V[0, :current_alignment_length]

        return Pi_V, max(alignment_lengths)

    def compute_Pi_L(self, CDR3_seq_nt, Pi_V, max_V_align):
        """
        计算所有可能的Vgene
        Parameters
        ----------

        CDR3_seq_nt: String
            TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC
        V_usage_mask: list
            ["TRBV1", "TRBV2", "TRBV4", ... ,"TRBV30"]
            可用的Vgene 列表
        max_V_align: int
            最大匹配长度
        max_insertions: int
            最大插入
        PinsVD: list
            1*max_insertions,插入碱基个数分布
        Rvd: ndarray
            4*4转移矩阵
        Returns
        -------
        Pi_L : ndarray

            长度等于CDR3_seq_nt的一个一维数组
            Pi_L[i] 表示 CDR3序列的前i个碱基由V, Vdel, VDins 三个事件联合产生的概率
        大致流程就是先枚举每个位置i，再枚举每个Vgene V, 比对之后得出Vdel, 进而计算出VDins,
        然后根据IGOR模型的参数计算出概率然后相加
        """

        """
        碱基插入的思路：
        1. 对序列[0,最大匹配+最大插入]均考虑为有可能是插入
        2. 用插入碱基个数分布+转移矩阵进行计算
        """

        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

        max_insertions = len(self.PinsVD) - 1
        Pi_L = np.zeros((1, len(CDR3_seq_nt)))  ####引用时Pi_L要使用[0，x]不然都是错的
        for init_pos in range(0, max_V_align):
            ####Pi_L[init_pos] += Pi_V[init_pos] * self.PinsVD[0] can't multiply sequence by non-int
            Pi_L[0, init_pos] += self.PinsVD[0] * Pi_V[0, init_pos]
            for i in range(1, max_insertions):
                ####Pi_L[0, init_pos + i] += Pi_V[0, init_pos + i] * self.PinsVD[i] * self.Rvd[
                   ####nt2num[init_pos + i], nt2num[init_pos]] 明显写错了,Rvd是一个4*2的矩阵，行是actg，列是有关下一位的东西
                Pi_L[0, init_pos + i] += Pi_V[0, init_pos + i] * self.PinsVD[i] * self.Rvd[
                    nt2num[CDR3_seq_nt[init_pos + i]], nt2num[CDR3_seq_nt[init_pos]]]

        return Pi_L

    def compute_Pi_J_given_D(self, CDR3_seq_nt, J_usage_mask):
        """Compute Pi_J conditioned on D.
        This function returns the Pi array from the model factors of the D and J
        genomic contributions, P(D, J)*P(delJ|J) = P(D|J)P(J)P(delJ|J). This
        corresponds to J(D)^{x_1}.
        For clarity in parsing the algorithm implementation, we include which
        instance attributes are used in the method as 'parameters.'
        Parameters
        ----------
        CDR3_seq_nt : str
            Nucleotide sequence that input from the beginning

        J_usage_mask : list
            Indices of the J alleles to be considered in the Pgen computation.
        self.cutJ_genomic_CDR3_segs : list
            Indices of the J alleles to be considered in the Pgen computation.

        self.PD_given_J : ndarray
            Probability distribution of D conditioned on J, i.e. P(D|J).
            eg. For T-cells beta chains, IGoR deliver PD_given_J as a 3*15 matrix,
            representing 3 kinds of D genes and 15 kinds of J genes respectively.

        self.PJdelJ_nt_pos_vec : list of ndarrays
            For each J allele, format P(J)*P(delJ|J) into the correct form for
            a Pi array or J(D)^{x_4}. This is only done for the first and last
            position in each codon.

        self.PJdelJ_2nd_nt_pos_per_aa_vec : list of dicts
            For each J allele, and each 'amino acid', format P(J)*P(delJ|J) for
            positions in the middle of a codon into the correct form for a Pi
            array or J(D)^{x_4} given the 'amino acid'.
        Returns
        -------
        Pi_J_given_D : list
            List of (1, L) ndarrays corresponding to J(D)^{x_1}.
        max_J_align: int
            Maximum alignment of the CDR3_seq to any genomic J allele allowed by
            J_usage_mask.
        """

        # Note, the cutJ_genomic_CDR3_segs INCLUDE the palindromic insertions and thus are max_palindrome nts longer than the template.
        # furthermore, the genomic sequence should be pruned to start at a conserved region on the J side
        num_D_genes = self.PD_given_J.shape[0]  # 来自于process_generative_model...，T细胞beta chain 的这个数据是3
        Pi_J_given_D = [np.zeros((1, len(CDR3_seq_nt))) for i in
                        range(
                            num_D_genes)]  # Holds the aggregate weight for each nt possiblity and position每个D基因搞一个矩阵，用来放其对应的权重矩阵

        alignment_lengths = []
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        for J_in in J_usage_mask:  # 取出每个J_in对应的修剪基因
            try:
                cutJ_gen_seg = self.cutJ_genomic_CDR3_segs[J_in]  # 是一个nt序列
            except IndexError:
                print('Check provided V usage mask. Contains indicies out of allowed range.')
                continue
            current_alignment_length = self.max_nt_to_nt_alignment_right(CDR3_seq_nt, cutJ_gen_seg)
            alignment_lengths += [current_alignment_length]
            current_Pi_J = np.zeros((1, len(CDR3_seq_nt)))  # 临时储存结果的矩阵，待会还要去根据不同的D基因选择乘上P（D），这里只是初始的匹配上J基因片段后的概率

            if current_alignment_length > 0:
                for pos in range(current_alignment_length):
                    current_Pi_J[-(pos + 1)] = self.PJdelJ_nt_pos_vec[J_in][
                        nt2num[cutJ_gen_seg[-(pos + 1)]], -(pos + 1)]

                for pos in range(-2, -current_alignment_length - 1, -3):  # 注意有一点比较奇怪，他对超出alignment的第一个2nd condon也赋值了
                    condon = cutJ_gen_seg[pos - 1] + cutJ_gen_seg[pos + 1] + cutJ_gen_seg[
                        pos + 1]  # 应该序列的方向还是从5到3,所以构建Condon还是从左到右
                    current_Pi_J[0, pos] = self.PJdelJ_2nd_nt_pos_per_aa_vec[J_in][self.condons_dict[condon]][
                        nt2num[cutJ_gen_seg[pos]], pos]

            for D_in, pd_given_j in enumerate(
                    self.PD_given_J[:, J_in]):  # 这里的PD_given_J是一个3*15的矩阵，选中一个列J_in之后,D_in代表不同的J基因,pd_given_J是概率
                Pi_J_given_D[D_in][0, -current_alignment_length:] += pd_given_j * current_Pi_J[0,
                                                                                  -current_alignment_length:]

        return Pi_J_given_D, max(alignment_lengths)

    # Include DJ insertions (Rdj and PinsDJ), return Pi_JinsDJ_given_D
    def compute_Pi_JinsDJ_given_D(self, CDR3_seq_nt, Pi_J_given_D, max_J_align):
        """
        CDR3_seq_nt：序列
        Pi_J_given_D：基础概率
        max_J_align：最大匹配，限制插入

        Return:
        Pi_JinsDJ_given_D[D_in, len(CDR3_seq_nt)]
        """
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

        max_insertions = len(self.PinsDJ) - 1
        Pi_JinsDJ_given_D = [np.zeros((len(CDR3_seq_nt))) for i in range(len(Pi_J_given_D))]

        for D_in in range(len(Pi_J_given_D)):
            for init_pos in range(-1, -(max_J_align + 1), -1):
                Pi_JinsDJ_given_D[D_in][:, init_pos] += self.PinsDJ[0] * Pi_J_given_D[D_in][:, init_pos]
                for i in CDR3_seq_nt[init_pos - 1: init_pos - max_insertions: -1]:
                    Pi_JinsDJ_given_D[init_pos + i] += Pi_J_given_D[D_in, init_pos + i] * self.PinsVD[i] * self.Rdj[
                        nt2num[init_pos + i], nt2num[init_pos]]

        return Pi_JinsDJ_given_D

    def compute_Pi_R(self, CDR3_seq_nt, Pi_JinsDJ_given_D):

        '''
        CDR3_seq_nt: "ACTATCTGTGGTACT"
        Pi_JinsDJ_given_D: [D_in, 3L],这个可以根据之后函数改名字，基础概率
        思路：
        1. 将基础概率加入空矩阵
        2. 将删除概率乘进去,删除概率与D基因选取，D左右删除个数有关
        需要遍历的：
        1. D_in
        2. 位置pos（认为从不同位置开始删除），左边删除个数 （delDl）
        3. delDr（右边删除的个数）

        PD_nt_pos_vec PD_2nd_nt_pos_per_aa_vec : 没啥用，确定谁到谁为1，其他为0,PD_2nd_nt_pos_per_aa_vec确认了下这三个nt可以组成aa
        用PdelDldelDr_given_D算删除的
        重要参数：
        PdelDldelDr_given_D[delDl,delDr,D_in],[int,int,int],[左删除个数，右删除个数，D基因选择]
        Pi_JinsDJ_given_D[D_in, 3L],[int,int]

        Returns
        -------
        Pi_R : ndarray
            长度等于CDR3_seq_nt的一个一维数组
            Pi_R[i] 表示 CDR3序列的前i个碱基由J, Jdel, JDins, Ddel四个事件联合产生的概率

        还没考虑的：
        可能报错的东西没考虑完
        '''
        nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        Pi_R = np.zeros((len(CDR3_seq_nt)))
        min_pos = -len(CDR3_seq_nt)

        # 最大删除
        num_dell_pos, num_delr_pos, num_D_genes = self.PdelDldelDr_given_D.shape

        # 将之前算的加进来
        base_prob = Pi_JinsDJ_given_D

        # 循环算删除的
        for D_in, cutD_gen_seg in enumerate(self.cutD_genomic_CDR3_segs):
            l_D_seg = len(cutD_gen_seg)
            ####for init_pos in range(-1, -len(CDR3_seq_nt) * 3 - 1, -3):
            for init_pos in range(-1, -len(CDR3_seq_nt) - 1, -3):
                # 先算D基因全部删除的
                ####Pi_R[init_pos] += Pi_JinsDJ_given_D[D_in][:, init_pos] * self.zeroD_given_D[D_in] * base_prob[init_pos]
                ####IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

                Pi_R[init_pos] += Pi_JinsDJ_given_D[D_in][init_pos] * self.zeroD_given_D[D_in] * base_prob[D_in][init_pos]
                for delDr in range(num_delr_pos):
                    for pos in range(init_pos - 1, max(init_pos - l_D_seg + delDr, min_pos) - 1, -1):
                        # 用pos去算D基因左边删除的个数，并遍历delDl
                        D_pos = l_D_seg - delDr - 1 - ((init_pos - 1) - pos)
                        if D_pos > self.max_delDl_given_DdelDr[D_in][delDr]:
                            current_PdelDldelDr = 0
                        else:
                            current_PdelDldelDr = self.PdelDldelDr_given_D[D_pos, delDr, D_in]

                        ####Pi_R[pos] += current_PdelDldelDr * Pi_JinsDJ_given_D[D_in, pos] * base_prob[init_pos] baseprob是一个3*L的矩阵
                        Pi_R[pos] += current_PdelDldelDr * Pi_JinsDJ_given_D[D_in][pos] * base_prob[D_in][init_pos]
        return Pi_R

    def Probability(self, CDR3_seq_nt):
        """
        计算所有可能的Vgene
        Parameters
        ----------
        CDR3_seq: String
            TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC
        Returns
        -------
        pgen:  Float
            输入序列的生成概率
        """

        """
        我看原代码的算法是将输入序列拆解成两部分Left和Right, 然后做的概率相加
        Left部分由V，Vdel 和 VDins 产生
        Right部分由 DJ, Jdel, DJins 和, delD (3' 5') 产生
        """

        Pi_L = self.compute_Pi_L(CDR3_seq_nt, self.V_usage_mask)

        Pi_R = self.compute_Pi_R(CDR3_seq_nt, self.J_usage_mask)

        pgen = 0

        # zip Pi_L and Pi_R together to get total pgen
        for pos in range(len(CDR3_seq_nt) - 1):
            pgen += Pi_L[pos] * Pi_R[pos + 1]
        return pgen

    def compute_CDR3_pgen(self, CDR3_seq_nt, V_usage_mask, J_usage_mask):
        """Compute Pgen for CDR3 'amino acid' sequence CDR3_seq from VDJ model.

        Conditioned on the already formatted V genes/alleles indicated in
        V_usage_mask and the J genes/alleles in J_usage_mask.
        (Examples are TCRB sequences/model)

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        V_usage_mask : list
            Indices of the V alleles to be considered in the Pgen computation
        J_usage_mask : list
            Indices of the J alleles to be considered in the Pgen computation

        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence

        Examples
        --------
        >>> compute_CDR3_pgen('CAWSVAPDRGGYTF', ppp, [42], [1])
        1.203646865765782e-10
        >>> compute_CDR3_pgen(nt2codon_rep('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC'), ppp, [42], [1])
        2.3986503758867323e-12
        >>> compute_CDR3_pgen('\xbb\x96\xab\xb8\x8e\xb6\xa5\x92\xa8\xba\x9a\x93\x94\x9f', ppp, [42], [1])
        2.3986503758867323e-12

        """

        # Genomic V alignment/matching (contribution from P(V, delV)), return Pi_V
        Pi_V, max_V_align = self.compute_Pi_V_nt(CDR3_seq_nt, V_usage_mask)

        # Include VD insertions (Rvd and PinsVD) to get the total contribution from the left (3') side. Return Pi_L
        Pi_L = self.compute_Pi_L(CDR3_seq_nt, Pi_V, max_V_align)

        # Genomic J alignment/matching (contribution from P(D, J, delJ)), return Pi_J_given_D
        Pi_J_given_D, max_J_align = self.compute_Pi_J_given_D(CDR3_seq_nt, J_usage_mask)

        # Include DJ insertions (Rdj and PinsDJ), return Pi_JinsDJ_given_D
        Pi_JinsDJ_given_D = self.compute_Pi_JinsDJ_given_D(CDR3_seq_nt, Pi_J_given_D, max_J_align)

        # Include D genomic contribution (P(delDl, delDr | D)) to complete the contribution from the right (5') side. Return Pi_R
        Pi_R = self.compute_Pi_R(CDR3_seq_nt, Pi_JinsDJ_given_D)

        pgen = 0

        # zip Pi_L and Pi_R together to get total pgen

        for pos in range(len(CDR3_seq_nt) - 1):
            pgen += Pi_L[0][pos] * Pi_R[pos + 1] ####Pi_L和R性状有些不统一，但是能用

        return pgen

'''
    def compute_CDR3_pgen(self, CDR3_seq_nt, V_usage_mask, J_usage_mask):
        """Compute Pgen for CDR3 'amino acid' sequence CDR3_seq from VDJ model.

        Conditioned on the already formatted V genes/alleles indicated in 
        V_usage_mask and the J genes/alleles in J_usage_mask. 
        (Examples are TCRB sequences/model)

        Parameters
        ----------
        CDR3_seq : str
            CDR3 sequence composed of 'amino acids' (single character symbols
            each corresponding to a collection of codons as given by codons_dict).
        V_usage_mask : list
            Indices of the V alleles to be considered in the Pgen computation
        J_usage_mask : list
            Indices of the J alleles to be considered in the Pgen computation

        Returns
        -------
        pgen : float
            The generation probability (Pgen) of the sequence

        Examples
        --------
        >>> compute_CDR3_pgen('CAWSVAPDRGGYTF', ppp, [42], [1])
        1.203646865765782e-10
        >>> compute_CDR3_pgen(nt2codon_rep('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC'), ppp, [42], [1])
        2.3986503758867323e-12
        >>> compute_CDR3_pgen('\xbb\x96\xab\xb8\x8e\xb6\xa5\x92\xa8\xba\x9a\x93\x94\x9f', ppp, [42], [1])
        2.3986503758867323e-12

        """

        # Genomic V alignment/matching (contribution from P(V, delV)), return Pi_V
        Pi_V, max_V_align = self.compute_Pi_V(CDR3_seq_nt, V_usage_mask)

        # Include VD insertions (Rvd and PinsVD) to get the total contribution from the left (3') side. Return Pi_L
        Pi_L = self.compute_Pi_L(CDR3_seq, Pi_V, max_V_align)

        # Genomic J alignment/matching (contribution from P(D, J, delJ)), return Pi_J_given_D
        Pi_J_given_D, max_J_align = self.compute_Pi_J_given_D(CDR3_seq, J_usage_mask)

        # Include DJ insertions (Rdj and PinsDJ), return Pi_JinsDJ_given_D
        Pi_JinsDJ_given_D = self.compute_Pi_JinsDJ_given_D(CDR3_seq, Pi_J_given_D, max_J_align)

        # Include D genomic contribution (P(delDl, delDr | D)) to complete the contribution from the right (5') side. Return Pi_R
        Pi_R = self.compute_Pi_R(CDR3_seq, Pi_JinsDJ_given_D)

        pgen = 0

        # zip Pi_L and Pi_R together to get total pgen
        for pos in range(len(CDR3_seq) * 3 - 1):
            pgen += np.dot(Pi_L[:, pos], Pi_R[:, pos + 1])

        return pgen
'''

def CommandLineParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', help='Sequence to infer generation probability')
    return parser.parse_args()


if __name__ == "__main__":
    args = CommandLineParser()
    # args = { arg:getattr(args, arg) for arg in vars(args) }
    print("Input Sequence: ", args.infer)

    #准备GenomicData
    GD = load_model.GenomicData()
    GDVDJ = load_model.GenomicDataVDJ()
    #准备GenomicModel
    GMVDJ = load_model.GenerativeModelVDJ()

    ##加载GenomicDataVDJ里的参数
    GDVDJ.load_igor_genomic_data(
        "D:\BIO INFOR\OLGA\OLGA-master\olga\default_models\human_T_beta\model_params.txt",
        "D:\BIO INFOR\OLGA\OLGA-master\olga\default_models\human_T_beta\V_gene_CDR3_anchors.csv",
        "D:\BIO INFOR\OLGA\OLGA-master\olga\default_models\human_T_beta\J_gene_CDR3_anchors.csv")

    ##加载GenerativeModelVDJ里的参数
    GMVDJ.load_and_process_igor_model(
        "D:\BIO INFOR\OLGA\OLGA-master\olga\default_models\human_T_beta\model_marginals.txt")

    #准备PreprocessedParametersVDJ

    PPVDJ = PreprocessedParametersVDJ(GMVDJ, GDVDJ)

    '''
    验证报错中的问题
    if GMVDJ.__class__.__name__.endswith('VDJ') and GDVDJ.__class__.__name__.endswith('VDJ'):
        recomb_type = 'VDJ'
        print(recomb_type)
    else:
        raise ValueError  # recomb types must match
    '''
    #准备当前要使用的GenerationProbability
    GP = GenerationProbability()
    GPVDJ = GenerationProbabilityVDJ(GMVDJ, GDVDJ)  #初始化还是GM和GD
    #print(PPVDJ.d_V_usage_mask)
    #print(GP.codons_dict)
    print(GPVDJ.compute_CDR3_pgen('CAWSVAPDRGGYTF', [42], [1]))
    #pgen_model.compute_nt_CDR3_pgen('TGTGCCAGTAGTATAACAACCCAGGGCTTGTACGAGCAGTACTTC')

