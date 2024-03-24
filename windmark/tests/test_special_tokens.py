from windmark.core.structs import Tokens


def test_token_indices_in_succeeding_order():
    assert int(max(Tokens.__members__.values())) + 1 == len(Tokens)


def test_token_indices_are_not_negative():
    for index in Tokens.__members__.values():
        assert index >= 0
