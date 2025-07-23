class ComputeStrategyBase():
    def __init__(self, strategy_number):
        '''
            Base class to get the setup of the machines for the distributed training.
            Definitions:
                - N: Number of machines for the distributed training (initial input).
                - S: Number of spot instances.
                - D: Number of on-demand instances. Minimum value of needed CPUs to fit model in memory.
        '''
        self.strategy_number = strategy_number

    def _get_N():
        pass

    def _get_S():
        pass

    def _get_D():
        pass


class ComputeStrategy1(ComputeStrategyBase):
    '''
        Setup: N=D, Total=N
    '''

    def _get_N(self, N, D, S):
        return N

    def _get_S(self, N, D):
        return 0

    def _get_D(self, N, D):
        return N


class ComputeStrategy2(ComputeStrategyBase):
    '''
        Setup: N=D+S, Total=N
        If N=D, we do not have S (this is case 1)
    '''

    def _get_N(self, N, D, S):
        return D+S

    def _get_S(self, N, D):
        S = max(N - D, 0)
        # print(N,D,S)
        return int(S)

    def _get_D(self, N, D):
        return D


class ComputeStrategy3(ComputeStrategyBase):
    '''
        Setup: N=S, Total=D+N
    '''

    def _get_N(self, N, D, S):
        return N

    def _get_S(self, N, D):
        return N

    def _get_D(self, N, D):
        return D


class ComputeStrategy4(ComputeStrategyBase):
    '''
        Setup: N=S, Total=S
    '''

    def _get_N(self, N, D, S):
        return N

    def _get_S(self, N, D):
        return N

    def _get_D(self, N, D):
        return 0


def get_strategy_class(strategy):
    strategy_mapping = {
        1: ComputeStrategy1,
        2: ComputeStrategy2,
        3: ComputeStrategy3,
        4: ComputeStrategy4
    }
    return strategy_mapping[strategy]
