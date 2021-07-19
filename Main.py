from GeneticAlgorithm import *
from Parser import *


def main():
    data_Lcopy = ["aa", "bb", "aaaa", "abab", "baba", "bbbb", "aaaaaa", "aabaab", "abaaba", "baabaa", "abbabb",
                  "bbabba", "bbbbbb", "aaaaaaaa", "aaabaaab", "aabaaaba", "abaaabaa", "baaabaaa", "aabbaabb",
                  "abbaabba", "bbaabbaa", "abbbabbb", "bbbabbba", "abababab", "babababa",
                  "bbbbbbbb", "aaaaaaaaaa", "aaaabaaaab", "aaabaaaaba", "aabaaaabaa", "abaaaabaaa", "baaaabaaaa", "aaabbaaabb",
                  "aabbaaabba", "abbaaabbaa", "bbaaabbaaa", "aabbbaabbb", "abbbaabbba","bbbaabbbaa", "abbbbabbbb",
                  ]
    data_Lpal = ["a", "b", "aa", "bb", "aaa", "aba", "bab", "bbb", "aaaa", "abba", "baab", "bbbb",
                 "aabaa", "baaab", "abbba", "abaaba", "abbabba",
                 "abbaabba", "abababa", "aabaaabaa", "babab", "babbab", "abaaaba", "aaabaaa", "aabbbaa", "abbbba", "babbbbab"]

    data_Lanbn = ["ab", "aabb", "aaabbb", "aaaabbbb", "aaaaabbbbb"]

    Gcon = Grammar(["a", "b"], ["P", "S"], ["X", "Y"],
                   [
                       PRule(Pred("P", ["a"]), None),
                       PRule(Pred("P", ["b"]), None),
                       PRule(Pred("P", ["XY"]), [Pred("P", ["X"]), Pred("P", ["Y"])]),
                       PRule(Pred("S", ["X"]), [Pred("P", ["X"])])
                   ])

    GA = GeneticAlgorithm(steps=10000, pop_size=200, penalty=35, mortality=.5, mutation=.6)
    result = GA.run(Gcon, data_Lanbn)
    print(result)


main()