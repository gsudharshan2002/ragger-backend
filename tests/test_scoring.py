from app.services.evaluation.scoring import is_refusal, keyword_present, passed_strict


def test_short_keyword_needs_whole_word():
    assert not keyword_present("I know it is not now", "no")
    assert keyword_present("There is no replacement field.", "no")


def test_stem_keyword_allows_suffix():
    assert keyword_present("You should be migrating to resources.core", "migrat")


def test_dotted_and_multiword_keywords():
    assert keyword_present("Read `resources.core` instead.", "resources.core")
    assert keyword_present("Use the repository's commit  history.", "commit history")


def test_refusal_that_restates_the_question_fails():
    # The false pass seen in Week 10 (gh-004): refusal containing the keyword.
    answer = (
        "The retrieved evidence does not contain information about the type "
        "value reported for a submodule entry."
    )
    assert is_refusal(answer)
    assert not passed_strict(answer, ["submodule"])


def test_real_answer_passes():
    assert passed_strict('A submodule entry reports "type": "submodule".', ["submodule"])


def test_all_keywords_required():
    assert not passed_strict("The field was removed.", ["no", "removed"])
    assert passed_strict("No, it was removed.", ["no", "removed"])


def test_correct_negative_answer_is_not_a_refusal():
    answer = "No. The top-level rate property was removed; the response does not include it."
    assert not is_refusal(answer)
    assert passed_strict(answer, ["no", "removed"])
