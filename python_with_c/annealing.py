def T_anneal(T, ii, num_steps, num_burnin):

    # Implement annealing code here

    # Geometric annealing
    # T_start = 4
    # a = 0.9997
    # T_a = T + (T_start - T) * (a ** ii) if ii < num_burnin else T


    # Linear annealing
    T_start = 4
    T_a = T_start + (T - T_start) * float(ii) / float(num_burnin) if ii < num_burnin else T

    # No annealing
    # T_a = T

    return float(T_a)


def B_anneal(B, ii, num_steps, num_burnin):

    # Implement annealing code here

    # Geometric annealing
    # B_start = 0.1
    # a = 0.9997
    # B_a = B + B_start * (a ** ii) if ii < num_burnin else B

    # Linear annealing
    B_start = 0.1
    B_a = B_start + (B - B_start) * float(ii) / float(num_burnin) if ii < num_burnin else B

    # No annealing
    # B_a = B

    return float(B_a)
