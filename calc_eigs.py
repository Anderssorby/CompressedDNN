from odin.utils.default import default_chainer
import odin.plot as oplt

if __name__ == "__main__":
    co, args, model_wrapper = default_chainer()

    co.calc_inter_layer_covariance(model_wrapper=model_wrapper)

    datastore = model_wrapper.get_group("inter_layer_covariance")
    cov = datastore["cov"]
    eigen_values = datastore["eigen_values"]

    for i, eigs in enumerate(eigen_values):
        l = len(eigs)
        eigs = abs(eigs)
        oplt.plot_eigen_values(eigs, title="%s layer %d (%d)" % (model_wrapper.model_name, i, l))
        oplt.save("eigs(%d)" % i)
        oplt.plot_matrix(cov[i], title="Covariance (%d)" % (i))
        oplt.save("cov(%d)" % i)




