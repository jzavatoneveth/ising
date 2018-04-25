def T_anneal(T, T0, ii, num_steps, num_burnin):

    # Implement annealing code here

    # Linear annealing

    return float(float(T0) + float(T - T0) * float(ii) / float(num_burnin) if (ii < num_burnin and T < T0) else T)


def B_anneal(B, B0, ii, num_steps, num_burnin):

    # Implement annealing code here

    # Linear annealing

    return float(float(B0) + float(B - B0) * float(ii) / float(num_burnin) if (ii < num_burnin and B < B0) else B)
