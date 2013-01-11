import numpy

def compute_aco(zam_protein):
    return zam_protein.MeanContactOrder()

def iter_contacts(zam_protein):
    # contacts are 0-indexed
    res_contact_list = zam_protein.ResContactList()
    for c in res_contact_list:
        yield c

def compute_type2_contact_pairs(zam_protein):
    c2_list = []
    for c in iter_contacts(zam_protein):
        residue_i = c[0]
        residue_j = c[1]
        residue_m_bounds = range(residue_i, residue_j)
        m_contact_list = zam_protein.ResContactList(ResInd=residue_m_bounds)
        for c2 in m_contact_list:
            c_and_c2_are_same_contact = (c == c2)
            i_and_m_are_equal = (c[0] == c2[0])
            j_and_n_are_equal = (c[1] == c2[1])
            if c_and_c2_are_same_contact:
                continue
            elif i_and_m_are_equal and j_and_n_are_equal:
                continue
            elif i_and_m_are_equal and not j_and_n_are_equal:
                c2_list.append( (c, c2) )
            elif not i_and_m_are_equal and j_and_n_are_equal:
                c2_list.append( (c, c2) )
            else:
                c2_list.append( (c, c2) )
    return c2_list

def compute_type3_contact_pairs(zam_protein):
    c3_list = []
    for c in iter_contacts(zam_protein):
        residue_i = c[0]
        residue_j = c[1]
        residue_m_bounds = range(residue_i+1, len(zam_protein))
        m_contact_list = zam_protein.ResContactList(ResInd=residue_m_bounds)
        for c3 in m_contact_list:
            c_and_c3_are_same_contact = (c == c3)
            m_greater_than_or_equal_to_j = (c3[0] >= c[1])
            n_less_than_or_equal_to_j = (c3[1] <= c[1])
            i_and_m_are_equal = (c[0] == c3[0])
            j_and_n_are_equal = (c[1] == c3[1])
            if c_and_c3_are_same_contact:
                continue
            elif m_greater_than_or_equal_to_j:
                continue
            elif n_less_than_or_equal_to_j:
                continue
            elif i_and_m_are_equal or j_and_n_are_equal:
                continue
            else:
                c3_list.append( (c, c3) )
    return c3_list

def compute_coc1(zam_protein):
    seq_sep = []
    num_contacts = len(zam_protein.ResContactList())
    for c in iter_contacts(zam_protein):
        seq_sep.append(c[1] - c[0])
    seq_sep_array = numpy.array(seq_sep)
    coc1 = numpy.sum(seq_sep_array**2)
    coc1 /= (3. * num_contacts**2)
    return coc1

def compute_coc2(zam_protein):
    num_contacts = len(zam_protein.ResContactList())
    c2_list = compute_type2_contact_pairs(zam_protein)
    c3_list = compute_type3_contact_pairs(zam_protein)
    c3_seq_sep = []
    for c3 in c3_list:
        ij = c3[0]
        j = ij[1]
        mn = c3[1]
        m = mn[0]
        c3_seq_sep.append(j - m)
    c2_seq_sep = []
    for c2 in c2_list:
        mn = c2[1]
        m = mn[0]
        n = mn[1]
        c2_seq_sep.append(n - m)
    c3_seq_sep_array = numpy.array(c3_seq_sep)
    c2_seq_sep_array = numpy.array(c2_seq_sep)
    coc2 = numpy.sum(c3_seq_sep_array**2) + numpy.sum(c2_seq_sep_array**2)
    coc2 /= (3. * num_contacts**2)
    coc2 *= 2
    return coc2
