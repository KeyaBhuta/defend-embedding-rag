"""Future-training interface, NOT an SFT/RL trainer."""
def process_rewards(audit):
    """Human-calibrated verifier scores required before using for optimization."""
    return {stage: audit[stage]/2 for stage in ('dp','etp','rap','synthesis')}

def discounted_return(rewards, gamma=1.0):
    if not 0 <= gamma <= 1: raise ValueError('gamma must be in [0,1]')
    total=0.0
    returns=[]
    for reward in reversed(rewards):
        total=reward+gamma*total
        returns.append(total)
    return list(reversed(returns))
