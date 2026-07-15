from wandb_query import format_results


def test_format_results_default():
    averages = {"pong": 15.0, "alien": 150.0}
    assert format_results(averages) == "alien: 150.00\npong: 15.00"


def test_format_results_bare():
    averages = {"pong": 15.0, "alien": 150.0}
    assert format_results(averages, bare=True) == "150.00\n15.00"


def test_format_results_empty():
    assert format_results({}) == ""


def test_format_results_single_default():
    assert format_results({"pong": 42.5}) == "pong: 42.50"


def test_format_results_single_bare():
    assert format_results({"pong": 42.5}, bare=True) == "42.50"


def test_format_results_alphabetical_order():
    averages = {"zebra": 1.0, "aardvark": 2.0, "mango": 3.0}
    result = format_results(averages)
    assert result == "aardvark: 2.00\nmango: 3.00\nzebra: 1.00"


def test_format_results_bare_alphabetical_order():
    averages = {"zebra": 1.0, "aardvark": 2.0, "mango": 3.0}
    result = format_results(averages, bare=True)
    assert result == "2.00\n3.00\n1.00"
