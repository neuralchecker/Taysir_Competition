from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import ProbabilisticDeterministicFiniteAutomaton
from pythautomata.automata.fast_implementations.fast_pdfa import FastProbabilisticDeterministicFiniteAutomaton as FastPDFA
from pythautomata.automata.fast_implementations.faster_pdfa import FasterProbabilisticDeterministicFiniteAutomaton as FasterPDFA
from collections import OrderedDict


class FastProbabilisticDeterministicFiniteAutomatonConverter():
    """
    Class that transforms a pythautomata PDFA to a FastPDFA
    """

    def to_fast_pdfa(self, pdfa: ProbabilisticDeterministicFiniteAutomaton):
        alphabet = [int(x.value) for x in pdfa.alphabet]
        initial_state = list(pdfa.initial_states)[0].name
        transition_function = dict()
        probability_function = dict()
        terminal_symbol = int(pdfa.terminal_symbol.value)
        name = pdfa.name+"FAST"
        for state in pdfa.weighted_states:
            transition_function[state.name] = dict()
            probability_function[state.name] = OrderedDict()
            for transition_symbol in sorted(list(state.transitions_set.keys())):
                symbol = int(transition_symbol.value)
                transition_function[state.name][symbol] = list(
                    state.transitions_set[transition_symbol])[0].next_state.name
                probability_function[state.name][symbol] = list(
                    state.transitions_set[transition_symbol])[0].weight
            probability_function[state.name][terminal_symbol] = state.final_weight
        fast_pdfa = FastPDFA(alphabet, initial_state, transition_function,
                             probability_function, terminal_symbol, name)
        return fast_pdfa

    def to_faster_pdfa(self, pdfa: ProbabilisticDeterministicFiniteAutomaton):
        alphabet = [int(x.value) for x in pdfa.alphabet.symbols]
        initial_state = list(pdfa.initial_states)[0].name
        transition_function = dict()
        probability_function = dict()
        terminal_symbol = int(pdfa.terminal_symbol.value)
        name = pdfa.name+"FASTER"
        for state in pdfa.weighted_states:
            transition_function[state.name] = dict()
            probability_function[state.name] = OrderedDict()
            for transition_symbol in sorted(list(state.transitions_set.keys())):
                symbol = int(transition_symbol.value)
                transition_function[state.name][symbol] = list(
                    state.transitions_set[transition_symbol])[0].next_state.name
                probability_function[state.name][symbol] = list(
                    state.transitions_set[transition_symbol])[0].weight
            probability_function[state.name][terminal_symbol] = state.final_weight
        fast_pdfa = FasterPDFA(alphabet, initial_state, transition_function,
                               probability_function, terminal_symbol, name)
        return fast_pdfa
