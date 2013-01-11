import nose.tools
from foldkin.zam_protein import create_zam_protein
from foldkin.contact_order import compute_aco, compute_coc1, compute_coc2

EPSILON = 0.01
PROTG_PATH = 'foldkin/test/test_data/1PGB.pdb'
EXPECTED_PROTG_ACO = 17.02
EXPECTED_PROTG_COC1 = 1.23
EXPECTED_PROTG_COC2 = 35.17

@nose.tools.istest
def test_computes_correct_aco():
    """docstring for test_computes_correct_aco"""
    zam_protein = create_zam_protein(PROTG_PATH)
    aco = compute_aco(zam_protein)
    error_msg = "Expected %.2f, got %.2f" % (EXPECTED_PROTG_ACO, aco)
    aco_diff = abs(aco - EXPECTED_PROTG_ACO)
    nose.tools.ok_(aco_diff < EPSILON, error_msg)

@nose.tools.istest
def test_computes_correct_coc1():
    """docstring for test_computes_correct_coc1"""
    zam_protein = create_zam_protein(PROTG_PATH)
    coc1 = compute_coc1(zam_protein)
    error_msg = "Expected %.2f, got %.2f" % (EXPECTED_PROTG_COC1, coc1)
    coc1_diff = abs(coc1 - EXPECTED_PROTG_COC1)
    nose.tools.ok_(coc1_diff < EPSILON, error_msg)

@nose.tools.istest
def test_computes_correct_coc2():
    """docstring for test_computes_correct_coc2"""
    zam_protein = create_zam_protein(PROTG_PATH)
    coc2 = compute_coc2(zam_protein)
    error_msg = "Expected %.2f, got %.2f" % (EXPECTED_PROTG_COC2, coc2)
    coc2_diff = abs(coc2 - EXPECTED_PROTG_COC2)
    nose.tools.ok_(coc2_diff < EPSILON, error_msg)
