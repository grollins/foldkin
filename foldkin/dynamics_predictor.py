import numpy
import pylab
import ratespec.RateSpecClass
import ratespec.RateSpecTools
import ratespec.PlottingTools
from base.data_predictor import DataPredictor
from foldkin.fold_rate_prediction import FoldRatePrediction
from foldkin.linalg import ScipyMatrixExponential, vector_matrix_product
from foldkin.probability_vector import VectorTrajectory

class DynamicFoldRatePredictor(DataPredictor):
    """docstring for DynamicFoldRatePredictor"""
    def __init__(self, make_plots, noisy=False):
        super(DynamicFoldRatePredictor, self).__init__()
        self.expm = ScipyMatrixExponential()
        self.prediction_factory = FoldRatePrediction
        self.make_plots = make_plots
        self.noisy = noisy

    def predict_data(self, model, feature=None):
        return self.predict_fold_rate_from_dynamics(model)

    def predict_fold_rate_from_dynamics(self, model):
        # =====================================================
        # = 0. Get rate matrix and initial probability vector =
        # =====================================================
        init_prob_vec = model.get_init_prob_vec()
        Q = model.build_rate_matrix(0.0)

        if self.noisy:
            print init_prob_vec
            print Q

        # ====================================================
        # = 1. Run a coarse trajectory over a wide range     =
        # ====================================================
        broad_t_range = 10**numpy.arange(-15, 15, 1.0)
        broad_vector_trajectory = VectorTrajectory(model.state_id_list)
        for i,t in enumerate(broad_t_range):
            try:
                eQt = self.expm.compute_matrix_exp(Q, t)
            except ValueError:
                print "eQt not finite at %d, %.2e" % (i, t)
                break
            prob_vec_at_time_t = vector_matrix_product(
                                    init_prob_vec, eQt,
                                    do_alignment=True)
            broad_vector_trajectory.add_vector(prob_vec_at_time_t)

        # ========================
        # = 2. find the midpoint =
        # ========================
        folded_state_id = model.state_id_list[model.folded_index]
        boltzmann_factor_array = model.compute_boltzmann_factors()
        folded_state_at_equilibrium = boltzmann_factor_array[model.folded_index]
        folded_state_at_equilibrium /= boltzmann_factor_array.sum()
        midpt = folded_state_at_equilibrium / 2.
        min_ind = -1
        for ind, vec in enumerate(broad_vector_trajectory):
            this_folded_prob = vec.get_state_probability(folded_state_id)
            delta_prob = this_folded_prob - midpt
            if delta_prob >= 0.0:
                min_ind = ind
                break
        if min_ind == -1:
            print "Never found the midpoint."
            raise RuntimeError
        time_at_point_closest_to_midpt = numpy.log10(broad_t_range[min_ind])

        # ====================================================
        # = 3. Run a more focused trajectory around midpoint =
        # ====================================================
        min_time = time_at_point_closest_to_midpt - 3
        max_time = time_at_point_closest_to_midpt + 3
        time_range_around_midpt = 10**numpy.arange(min_time, max_time, 0.1)
        midpt_vector_trajectory = VectorTrajectory(model.state_id_list)
        for i,t in enumerate(time_range_around_midpt):
            eQt = self.expm.compute_matrix_exp(Q, t)
            prob_vec_at_time_t = vector_matrix_product(
                                    init_prob_vec, eQt,
                                    do_alignment=True)
            midpt_vector_trajectory.add_vector(prob_vec_at_time_t)
        max_time = time_range_around_midpt.max()
        min_time = time_range_around_midpt.min()
        max_log_rate = numpy.log10(1./min_time)
        min_log_rate = numpy.log10(1./max_time)

        trajectory_file = 'temp.txt'
        with open(trajectory_file, 'w') as f:
            for i, v in enumerate(midpt_vector_trajectory):
                t = time_range_around_midpt[i]
                fold_prob = v.get_state_probability(folded_state_id)
                f.write("%.2e  %.2e\n" % (t, fold_prob))

        # ================================================
        # = 4. Run ratespec on trajectory to estimate kf =
        # ================================================
        log10_kf = self.ratespec_fit(
                    trajectory_file,
                    plot_name='%s_ratespec' % "my",
                    min_log_rate=min_log_rate, max_log_rate=max_log_rate)
        prediction = self.prediction_factory(log10_kf)
        prediction.trajectory = midpt_vector_trajectory
        return prediction

    def ratespec_fit(self, traj_file, plot_name, min_log_rate=-7,
                     max_log_rate=7):
        infile = traj_file
        MCsteps = 25
        nRates = 100
        timeUnit = 0.
        minrate = 10**min_log_rate
        maxrate = 10**max_log_rate
        outname = plot_name
        UseLnorm = 'lasso'

        r = ratespec.RateSpecClass.RateSpecClass(
                dataFile=infile, timeUnit=timeUnit, nRates=nRates,
                RateRange=(minrate, maxrate), Lnorm=UseLnorm,
                scaleData=True, standardizeData=False,
                init_sigma=0.1, init_tau=1.0, dsigma=1.0e-2, dtau=1.0e-2)

        # sample the posterior distribution of possible rate spectra
        r.sampleSpectrum(nsteps=MCsteps, Verbose=False)

        # Calculate the spectrum with the current value of lambda (w)
        w = (r.sigma/r.tau)**2
        A, rss, ndof = ratespec.RateSpecTools.fitRateSpectrum(
                        r.Times, r.Data, r.Rates, w, Lnorm=r.Lnorm,
                        standardizeData=False)

        # Calculate the fitted time trace
        FitData = ratespec.RateSpecTools.SumSpectra(A, r.Rates, r.Times)
        if self.make_plots:
            # make a plot
            pylab.figure()
            pylab.subplot(2,1,1)
            ratespec.PlottingTools.TimeSeriesWithFit(
                                    r.Times, r.Data, FitData,
                                    markersize=8)
            pylab.subplot(2,1,2)
            ratespec.PlottingTools.TimescaleSpectrum(
                                    r.Timescales, A, timeunit='s')
            plot_file_name = "%s_traj_fit.png" % outname
            pylab.savefig(plot_file_name)
            print "Wrote plot: %s" % plot_file_name
            pylab.clf()
        finite_TS_inds = numpy.where(r.Timescales < numpy.inf)[0]
        finite_TS = r.Timescales[finite_TS_inds]
        abs_finite_A = numpy.abs(A[finite_TS_inds])
        min_timescale = finite_TS[abs_finite_A.argmax()]
        max_rate = 1./min_timescale
        return numpy.log10(max_rate)
        