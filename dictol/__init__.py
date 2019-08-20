__version__ = '0.1.1'
def demo():
    from dictol import SRC, ODL, DLSI, COPAR, LRSDL, utils
    # mini test: 5 training samples per class
    SRC.mini_test_unit()
    DLSI.mini_test_unit()
    COPAR.mini_test_unit()
    LRSDL.mini_test_unit_FDDL()
    LRSDL.mini_test_unit()

    # test: 15 training samples per class
    SRC.test_unit()
    DLSI.test_unit()
    COPAR.test_unit()
    LRSDL.test_unit_FDDL()
    LRSDL.test_unit()

