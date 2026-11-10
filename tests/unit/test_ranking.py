from max_ai.core.ranking import bm25, fuse, words


def test_words_split_names_and_drop_stopwords():
    assert words("exa_search: Search the Web") == ["exa", "search", "search", "the", "web"]
    assert words("Book a plane", {"a"}) == ["book", "plane"]
    assert words("dónde dormir") == ["dónde", "dormir"]


def test_bm25_rare_words_count_more():
    docs = [["search", "web"], ["search", "hotels"], ["send", "email"]]
    web, hotels, email = bm25(["search", "web"], docs)
    assert web > hotels > 0 and email == 0


def test_bm25_without_query_or_docs():
    assert bm25([], [["a"]]) == [0.0]
    assert bm25(["a"], []) == []


def test_fuse_rewards_agreement_between_rankings():
    assert fuse([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]
    assert fuse([]) == []
