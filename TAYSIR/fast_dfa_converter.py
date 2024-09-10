from pythautomata.automata.deterministic_finite_automaton import DeterministicFiniteAutomaton
from fast_dfa import FastDeterministicFiniteAutomaton as FastDFA

class FastDeterministicFiniteAutomatonConverter():
    """
    Class that transforms a pythautomata DFA to a FastDFA
    """

    def to_fast_dfa(self, dfa: DeterministicFiniteAutomaton):
        alphabet = [int(x.value) for x in dfa.alphabet]
        initial_state = dfa.initial_state.name
        transition_function = dict()
        terminal_states = set()
        name = dfa.name+"FAST"
        for state in dfa.states:
            if state.is_final:
                terminal_states.add(state.name)
            for symbol, next_state in state.transitions.items():
                symbol = int(symbol.value)
                next_state = list(next_state)[0].name
                transition_function[(state.name, symbol)] = next_state
                   
        return FastDFA(alphabet, initial_state, transition_function,
                             terminal_states, name)