import zfit

def build_model(obs, obs_left, obs_right, init_values):
    # signal component
    mu = zfit.Parameter("mu", init_values['mu'], obs_left, obs_right)
    sigma = zfit.Parameter("sigma", init_values['sigma'], 0.01, obs_right - obs_left)
    signal = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs, name='sig')

    # background component
    background = zfit.pdf.Uniform(obs_left, obs_right, obs=obs, name='bkgr')
    # lambd = zfit.Parameter("lambda", -0.01, -1, -0.000001)
    # background = zfit.pdf.Exponential(lambd, obs=obs)

    # combing sig and bkgr together
    fr = zfit.Parameter("fr", init_values['fr'], 0, 1)
    model = zfit.pdf.SumPDF([signal, background], fracs=fr)
    # n_bkg = zfit.Parameter('n_bkg', sum(ima_x))
    # n_sig = zfit.Parameter('n_sig', 1000)
    # gauss_extended = gauss.create_extended(n_sig)
    # exp_extended = exponential.create_extended(n_bkg)
    # uni_extended = uniform.create_extended(n_bkg)
    # model = zfit.pdf.SumPDF([gauss_extended, uni_extended])
    return model
