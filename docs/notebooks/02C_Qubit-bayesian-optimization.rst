Single spin: Bayesian optimization of a gate
============================================

This is similar to the 02B_Single_qubit_gate example, except that it
uses Bayesian optimization (Shahriari et al., 2016)
:cite:p:`shahriari2016taking` instead of gradient descent.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    from jax import Array
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.logger import Logger
    from paraqeet.measurement.fidelity import UnitaryFidelity
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.bayesian_optimizer import BayesianOptimizer
    from paraqeet.propagation import Expm
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import ConstantEnvelope
    from paraqeet.signal.iq_mixer import IQMixer

Setup
-----

We first set up the qubit system we want to control. We set the qubit
frequency :math:`\omega_q / 2 \pi` to be :math:`4.8` GHz and define the
Hamiltonian as

.. math:: H(t)=H_\text{drift}+H_c(t)= \frac{\omega_q}{2} \sigma_z + \Omega(t)\sigma_x, 

\ where :math:`\Omega(t)` will be supplied by the generator.

.. code:: ipython3

    freq_q = 4.8e9
    omega_q = 2 * np.pi * freq_q
    
    qubit_hamiltonian = QubitHamiltonian(
        frequency=Quantity(omega_q, 0.8 * omega_q, 1.2 * omega_q, unit="Hz", two_pi=True), drives=[]
    )
    model = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

For signal generation, we define a simple cosine shaped tone generator
:math:`A \cos(\omega t)`

.. code:: ipython3

    t_simu = 3e-9
    tone = ConstantEnvelope()
    tone.t_final.set_value(t_simu)
    gen = IQMixer(envelopes=[tone])

We can inspect the pre-defined parameters with

.. code:: ipython3

    params_gen = gen.get_parameters()

In this notebook, we would like to optimize the amplitude ``Amplitude``
and frequency ``lo_freq`` of the drive. We add a drive on the qubit.

.. code:: ipython3

    freq = 4.8e9 * 2 * np.pi
    sigma_x = qubit_hamiltonian.sigma_x
    drive = Drive(sigma_x, gen)
    qubit_hamiltonian.drives = [drive]
    model = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

Textbook values for implementing an :math:`X` rotation on this system at
a time :math:`T` would be :math:`\omega=\omega_q` and :math:`A=\pi/T`.
We use some offset from these values as an initial guess to demonstrate
the optimization procedure.

.. code:: ipython3

    params_gen[0].set_value(0.5 * np.pi / t_simu)
    params_gen[2].set_value(1.01 * freq)

We select a propagation method, piecewise constant exponentiation, and
configure an :math:`X`-gate as a target gate. Also, we initialize the
identity at time :math:`0`.

.. code:: ipython3

    times = np.array([0.0, t_simu])
    
    prop = Expm(eom_func=model.get_value, resolution=100e9, initial_state=np.identity(2))
    gate_fid = UnitaryFidelity(propagation_func=prop.get_value, propagation_and_gradient_func=None, gate=sigma_x)

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_simu, 301)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_14_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    print(f"Gate fidelity: {gate_fid.get_value(times)}")


.. parsed-literal::

    Gate fidelity: 0.008842872531522971


Custom logger implementation
----------------------------

We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone. We also use a custom logger class
to collect all samples that the optimizer takes

.. code:: ipython3

    samples = []
    
    
    class CustomLogger(Logger):
        """Custom logger class definition."""
    
        def log(self, params: list[Quantity], infid: Array):
            """Log the list of quantities and the fidelity.
    
            Parameters
            ----------
            params: list[Quantity]
                List of parameters of the system.
            infid: Array
                Inverse of fidelity.
    
            """
            samples.append((params[0].get_value(), params[1].get_value()))
    
    
    optmap = OptimizationMap()
    optmap.add(gen, [params_gen[0], params_gen[2], params_gen[3]])
    opt = BayesianOptimizer(measure_func=gate_fid.get_value, optimization_map=optmap, initial_samples=10, iterations=100)
    opt.logger = CustomLogger()

.. code:: ipython3

    opt.optimize(times)


.. parsed-literal::

    |   iter    |  target   |     0     |     1     |     2     |
    -------------------------------------------------------------
    | [39m1        [39m | [39m0.0036278[39m | [39m-0.165955[39m | [39m0.4406489[39m | [39m-0.999771[39m |
    | [39m2        [39m | [39m0.0001104[39m | [39m-0.395334[39m | [39m-0.706488[39m | [39m-0.815322[39m |
    | [39m3        [39m | [39m0.0003939[39m | [39m-0.627479[39m | [39m-0.308878[39m | [39m-0.206465[39m |
    | [35m4        [39m | [35m0.0094778[39m | [35m0.0776334[39m | [35m-0.161610[39m | [35m0.3704390[39m |
    | [39m5        [39m | [39m3.085e-06[39m | [39m-0.591095[39m | [39m0.7562348[39m | [39m-0.945224[39m |
    | [35m6        [39m | [35m0.2028943[39m | [35m0.3409350[39m | [35m-0.165390[39m | [35m0.1173796[39m |
    | [39m7        [39m | [39m0.0007805[39m | [39m-0.719226[39m | [39m-0.603797[39m | [39m0.6014891[39m |
    | [39m8        [39m | [39m0.0434133[39m | [39m0.9365231[39m | [39m-0.373151[39m | [39m0.3846452[39m |
    | [39m9        [39m | [39m1.729e-06[39m | [39m0.7527783[39m | [39m0.7892133[39m | [39m-0.829911[39m |
    | [39m10       [39m | [39m3.574e-06[39m | [39m-0.921890[39m | [39m-0.660339[39m | [39m0.7562850[39m |


.. parsed-literal::

    | [35m11       [39m | [35m0.2425734[39m | [35m0.3863518[39m | [35m-0.166299[39m | [35m0.0746278[39m |


.. parsed-literal::

    | [35m12       [39m | [35m0.3066583[39m | [35m0.4027489[39m | [35m-0.070437[39m | [35m-0.062772[39m |
    | [39m13       [39m | [39m0.1000863[39m | [39m0.5626991[39m | [39m-0.235444[39m | [39m-0.322011[39m |


.. parsed-literal::

    | [39m14       [39m | [39m0.1240102[39m | [39m0.5454712[39m | [39m0.0932292[39m | [39m0.0241167[39m |
    | [39m15       [39m | [39m0.2938039[39m | [39m0.2579886[39m | [39m-0.151104[39m | [39m-0.137715[39m |


.. parsed-literal::

    | [39m16       [39m | [39m0.2955032[39m | [39m0.2598865[39m | [39m-0.148306[39m | [39m-0.140559[39m |
    | [39m17       [39m | [39m0.2785474[39m | [39m0.1549411[39m | [39m0.1498233[39m | [39m-0.217921[39m |


.. parsed-literal::

    | [39m18       [39m | [39m0.0051731[39m | [39m-0.073092[39m | [39m0.6008264[39m | [39m-0.041727[39m |
    | [35m19       [39m | [35m0.5373705[39m | [35m0.1386107[39m | [35m0.0137562[39m | [35m-0.486849[39m |


.. parsed-literal::

    | [39m20       [39m | [39m0.3633793[39m | [39m0.1027192[39m | [39m-0.076982[39m | [39m-0.669758[39m |
    | [39m21       [39m | [39m0.0637430[39m | [39m0.2637873[39m | [39m0.1406780[39m | [39m-0.555237[39m |


.. parsed-literal::

    | [39m22       [39m | [39m0.0522654[39m | [39m0.0470878[39m | [39m-0.075446[39m | [39m-0.430826[39m |
    | [39m23       [39m | [39m0.0011528[39m | [39m0.9867827[39m | [39m-0.644633[39m | [39m0.8585118[39m |
    | [39m24       [39m | [39m6.894e-05[39m | [39m-0.472166[39m | [39m-0.869049[39m | [39m-0.553717[39m |
    | [39m25       [39m | [39m5.132e-08[39m | [39m-0.877018[39m | [39m-0.657881[39m | [39m0.0207137[39m |
    | [39m26       [39m | [39m0.0011949[39m | [39m-0.795254[39m | [39m-0.508318[39m | [39m0.9524448[39m |


.. parsed-literal::

    | [35m27       [39m | [35m0.5590701[39m | [35m0.1431811[39m | [35m-0.003723[39m | [35m-0.559337[39m |


.. parsed-literal::

    | [39m28       [39m | [39m0.3727635[39m | [39m0.2058624[39m | [39m-0.043312[39m | [39m-0.517609[39m |


.. parsed-literal::

    | [39m29       [39m | [39m0.3966277[39m | [39m0.0825405[39m | [39m0.0616688[39m | [39m-0.552897[39m |


.. parsed-literal::

    | [39m30       [39m | [39m0.4136614[39m | [39m0.1003338[39m | [39m-0.035671[39m | [39m-0.551026[39m |


.. parsed-literal::

    | [39m31       [39m | [39m0.5339144[39m | [39m0.1673675[39m | [39m0.0165512[39m | [39m-0.649423[39m |


.. parsed-literal::

    | [35m32       [39m | [35m0.6124068[39m | [35m0.2367937[39m | [35m-0.040245[39m | [35m-0.733500[39m |


.. parsed-literal::

    | [39m33       [39m | [39m0.3211367[39m | [39m0.1986800[39m | [39m0.0082263[39m | [39m-0.818897[39m |
    | [39m34       [39m | [39m0.0137368[39m | [39m0.3362308[39m | [39m-0.840353[39m | [39m-0.837670[39m |


.. parsed-literal::

    | [39m35       [39m | [39m0.2460532[39m | [39m0.2928300[39m | [39m-0.119270[39m | [39m-0.709660[39m |
    | [39m36       [39m | [39m0.1168021[39m | [39m-0.049565[39m | [39m-0.108954[39m | [39m-0.202153[39m |


.. parsed-literal::

    | [39m37       [39m | [39m0.5141011[39m | [39m0.2657757[39m | [39m0.0174219[39m | [39m-0.704422[39m |
    | [39m38       [39m | [39m0.5764846[39m | [39m0.2122271[39m | [39m-0.042962[39m | [39m-0.669358[39m |


.. parsed-literal::

    | [39m39       [39m | [39m0.3956524[39m | [39m0.1417220[39m | [39m0.0993403[39m | [39m-0.397644[39m |
    | [39m40       [39m | [39m3.035e-07[39m | [39m-0.831439[39m | [39m0.3584873[39m | [39m-0.030378[39m |


.. parsed-literal::

    | [39m41       [39m | [39m0.4364339[39m | [39m0.3355248[39m | [39m0.0520019[39m | [39m-0.253463[39m |


.. parsed-literal::

    | [39m42       [39m | [39m0.1312007[39m | [39m0.3847292[39m | [39m0.1981156[39m | [39m-0.214810[39m |


.. parsed-literal::

    | [39m43       [39m | [39m0.2610900[39m | [39m0.2391648[39m | [39m-0.000474[39m | [39m-0.320963[39m |
    | [39m44       [39m | [39m0.0008451[39m | [39m-0.324396[39m | [39m0.8964272[39m | [39m-0.695403[39m |
    | [35m45       [39m | [35m0.7703155[39m | [35m0.9743153[39m | [35m0.0468998[39m | [35m0.6726593[39m |


.. parsed-literal::

    | [39m46       [39m | [39m0.5792541[39m | [39m0.9821157[39m | [39m0.1255883[39m | [39m0.7204063[39m |


.. parsed-literal::

    | [39m47       [39m | [39m0.4854179[39m | [39m0.8848882[39m | [39m0.0139857[39m | [39m0.6906923[39m |
    | [39m48       [39m | [39m0.1162966[39m | [39m-0.455283[39m | [39m0.0556825[39m | [39m-0.294502[39m |


.. parsed-literal::

    | [39m49       [39m | [39m0.0005492[39m | [39m-0.896720[39m | [39m-0.199863[39m | [39m-0.487141[39m |


.. parsed-literal::

    | [35m50       [39m | [35m0.8434360[39m | [35m1.0      [39m | [35m0.0731378[39m | [35m0.5856685[39m |


.. parsed-literal::

    | [39m51       [39m | [39m0.4756298[39m | [39m1.0      [39m | [39m-0.025385[39m | [39m0.5800696[39m |


.. parsed-literal::

    | [39m52       [39m | [39m0.3522303[39m | [39m0.9336380[39m | [39m0.1339088[39m | [39m0.5843633[39m |


.. parsed-literal::

    | [39m53       [39m | [39m0.8009691[39m | [39m1.0      [39m | [39m0.0842420[39m | [39m0.6369940[39m |


.. parsed-literal::

    | [39m54       [39m | [39m0.2177734[39m | [39m1.0      [39m | [39m0.0069847[39m | [39m0.7652570[39m |


.. parsed-literal::

    | [35m55       [39m | [35m0.8796430[39m | [35m0.9635119[39m | [35m0.0466889[39m | [35m0.6114599[39m |
    | [39m56       [39m | [39m5.541e-05[39m | [39m-0.879484[39m | [39m-0.954218[39m | [39m0.7154813[39m |


.. parsed-literal::

    | [35m57       [39m | [35m0.9392248[39m | [35m0.9564116[39m | [35m0.0423778[39m | [35m0.5418760[39m |


.. parsed-literal::

    | [39m58       [39m | [39m0.8826403[39m | [39m0.9717767[39m | [39m0.0459873[39m | [39m0.4555968[39m |


.. parsed-literal::

    | [35m59       [39m | [35m0.9593122[39m | [35m0.8851160[39m | [35m0.0101834[39m | [35m0.4696790[39m |


.. parsed-literal::

    | [39m60       [39m | [39m0.8575817[39m | [39m0.8781596[39m | [39m0.0253189[39m | [39m0.3671445[39m |


.. parsed-literal::

    | [39m61       [39m | [39m0.6719517[39m | [39m0.9170180[39m | [39m-0.055604[39m | [39m0.4032423[39m |


.. parsed-literal::

    | [39m62       [39m | [39m0.5844410[39m | [39m0.9816713[39m | [39m0.1248721[39m | [39m0.7212348[39m |


.. parsed-literal::

    | [39m63       [39m | [39m0.9034961[39m | [39m0.7778790[39m | [39m0.0279627[39m | [39m0.4297197[39m |


.. parsed-literal::

    | [39m64       [39m | [39m0.2761779[39m | [39m0.8398392[39m | [39m0.1044075[39m | [39m0.4196583[39m |


.. parsed-literal::

    | [39m65       [39m | [39m0.0003793[39m | [39m-0.712240[39m | [39m0.6268556[39m | [39m0.2073338[39m |
    | [39m66       [39m | [39m0.0001554[39m | [39m0.5744876[39m | [39m0.9758150[39m | [39m0.6370951[39m |


.. parsed-literal::

    | [39m67       [39m | [39m0.0004002[39m | [39m0.3575027[39m | [39m-0.296749[39m | [39m-0.023575[39m |


.. parsed-literal::

    | [39m68       [39m | [39m0.6014330[39m | [39m0.7765967[39m | [39m-0.057606[39m | [39m0.4158530[39m |


.. parsed-literal::

    | [39m69       [39m | [39m0.8312231[39m | [39m0.7922471[39m | [39m0.0024097[39m | [39m0.5224359[39m |


.. parsed-literal::

    | [39m70       [39m | [39m0.8823406[39m | [39m0.6765071[39m | [39m0.0301289[39m | [39m0.4710920[39m |


.. parsed-literal::

    | [39m71       [39m | [39m0.6820388[39m | [39m0.6695648[39m | [39m0.0394229[39m | [39m0.3613728[39m |


.. parsed-literal::

    | [39m72       [39m | [39m0.6813709[39m | [39m0.9910682[39m | [39m0.0295843[39m | [39m0.3006294[39m |


.. parsed-literal::

    | [39m73       [39m | [39m0.7281863[39m | [39m0.6153518[39m | [39m0.0178641[39m | [39m0.5842212[39m |


.. parsed-literal::

    | [39m74       [39m | [39m0.4349023[39m | [39m0.5521343[39m | [39m0.0947130[39m | [39m0.4802972[39m |


.. parsed-literal::

    | [39m75       [39m | [39m0.1919352[39m | [39m0.6154761[39m | [39m-0.077313[39m | [39m0.5060826[39m |


.. parsed-literal::

    | [39m76       [39m | [39m0.6649850[39m | [39m0.6955734[39m | [39m0.0881400[39m | [39m0.5632729[39m |
    | [39m77       [39m | [39m0.1196956[39m | [39m0.4775626[39m | [39m0.1868229[39m | [39m0.1878511[39m |


.. parsed-literal::

    | [39m78       [39m | [39m0.7736701[39m | [39m0.8444544[39m | [39m-0.002527[39m | [39m0.2455184[39m |


.. parsed-literal::

    | [39m79       [39m | [39m0.1726257[39m | [39m0.9323864[39m | [39m0.0311770[39m | [39m0.1289437[39m |


.. parsed-literal::

    | [39m80       [39m | [39m0.6308703[39m | [39m0.5852912[39m | [39m0.0633755[39m | [39m0.7207908[39m |


.. parsed-literal::

    | [39m81       [39m | [39m0.8620536[39m | [39m0.7327830[39m | [39m-0.040436[39m | [39m0.2360373[39m |


.. parsed-literal::

    | [39m82       [39m | [39m0.3610334[39m | [39m0.7669465[39m | [39m-0.138269[39m | [39m0.1803362[39m |


.. parsed-literal::

    | [39m83       [39m | [39m0.2680901[39m | [39m0.7456379[39m | [39m0.0531964[39m | [39m0.2414917[39m |
    | [39m84       [39m | [39m0.0002128[39m | [39m-0.873608[39m | [39m-0.590417[39m | [39m-0.322908[39m |


.. parsed-literal::

    | [39m85       [39m | [39m0.8492311[39m | [39m0.7937192[39m | [39m-0.052520[39m | [39m0.2973935[39m |


.. parsed-literal::

    | [39m86       [39m | [39m0.6405572[39m | [39m0.6527114[39m | [39m-0.082214[39m | [39m0.2750349[39m |


.. parsed-literal::

    | [39m87       [39m | [39m0.2158292[39m | [39m0.6862140[39m | [39m-0.017719[39m | [39m0.6862239[39m |


.. parsed-literal::

    | [39m88       [39m | [39m0.6545552[39m | [39m0.4640107[39m | [39m0.0769393[39m | [39m0.6622457[39m |


.. parsed-literal::

    | [39m89       [39m | [39m0.0729920[39m | [39m0.5485769[39m | [39m0.1991330[39m | [39m0.6784366[39m |


.. parsed-literal::

    | [39m90       [39m | [39m0.0418708[39m | [39m0.4638825[39m | [39m-0.027108[39m | [39m0.7473997[39m |


.. parsed-literal::

    | [39m91       [39m | [39m0.7876427[39m | [39m0.8877786[39m | [39m-0.001146[39m | [39m0.5417888[39m |
    | [39m92       [39m | [39m0.0009172[39m | [39m-0.723588[39m | [39m0.4703690[39m | [39m-0.997180[39m |


.. parsed-literal::

    | [39m93       [39m | [39m0.8957059[39m | [39m0.9033507[39m | [39m-0.048855[39m | [39m0.2893118[39m |


.. parsed-literal::

    | [39m94       [39m | [39m0.3349450[39m | [39m0.8852940[39m | [39m0.1722478[39m | [39m0.8723699[39m |


.. parsed-literal::

    | [39m95       [39m | [39m0.4422511[39m | [39m0.3215106[39m | [39m0.1035222[39m | [39m0.5829254[39m |
    | [39m96       [39m | [39m0.0001670[39m | [39m-0.257414[39m | [39m0.9945973[39m | [39m0.9939409[39m |


.. parsed-literal::

    | [39m97       [39m | [39m0.4775091[39m | [39m1.0      [39m | [39m-0.110546[39m | [39m0.2569957[39m |
    | [39m98       [39m | [39m0.0001739[39m | [39m-0.765238[39m | [39m-0.898005[39m | [39m0.5169565[39m |


.. parsed-literal::

    | [39m99       [39m | [39m0.0165422[39m | [39m-0.227062[39m | [39m-0.174050[39m | [39m0.2779688[39m |


.. parsed-literal::

    | [39m100      [39m | [39m4.686e-05[39m | [39m0.1375422[39m | [39m-1.0     [39m | [39m0.9424497[39m |


.. parsed-literal::

    | [39m101      [39m | [39m0.8646517[39m | [39m0.7383023[39m | [39m0.0442397[39m | [39m0.4828521[39m |
    | [39m102      [39m | [39m0.0      [39m | [39m-1.0     [39m | [39m0.3872079[39m | [39m1.0      [39m |


.. parsed-literal::

    | [39m103      [39m | [39m0.0008088[39m | [39m1.0      [39m | [39m-1.0     [39m | [39m-0.262179[39m |
    | [39m104      [39m | [39m0.0001732[39m | [39m1.0      [39m | [39m1.0      [39m | [39m-0.070733[39m |


.. parsed-literal::

    | [39m105      [39m | [39m5.308e-05[39m | [39m0.0711411[39m | [39m0.2757976[39m | [39m0.7103456[39m |


.. parsed-literal::

    | [39m106      [39m | [39m0.0080969[39m | [39m1.0      [39m | [39m-0.363937[39m | [39m-1.0     [39m |


.. parsed-literal::

    | [39m107      [39m | [39m4.781e-05[39m | [39m0.0270387[39m | [39m-1.0     [39m | [39m0.1049451[39m |
    | [39m108      [39m | [39m8.685e-05[39m | [39m-0.454330[39m | [39m0.6388374[39m | [39m-0.363639[39m |


.. parsed-literal::

    | [39m109      [39m | [39m0.0      [39m | [39m-1.0     [39m | [39m1.0      [39m | [39m0.7254970[39m |


.. parsed-literal::

    | [39m110      [39m | [39m0.0      [39m | [39m-1.0     [39m | [39m-1.0     [39m | [39m-1.0     [39m |
    =============================================================




.. parsed-literal::

    {'status': 0, 'value': 0.040687709101412284, 'iterations': 110, 'evaluations': 110}



The plot shows all the samples that the optimization took in the
two-dimensional parameter space. The red dot marks the best value.

.. code:: ipython3

    plt.figure(figsize=(4, 4))
    plt.scatter([s[0] for s in samples[:-1]], [s[1] for s in samples[:-1]], c="blue")
    plt.scatter([params_gen[0].get_value()], [params_gen[2].get_value()], c="red", marker="o", s=100)
    plt.xlim(params_gen[0].get_min_value()[0], params_gen[0].get_max_value()[0])
    plt.ylim(params_gen[2].get_min_value()[0], params_gen[2].get_max_value()[0])
    plt.xlabel(params_gen[0].get_name())
    plt.ylabel(params_gen[2].get_name())
    plt.show()



.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_21_0.png


.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>, <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_22_1.png


We can see from the plot and optimizer output that we have found good
controls.

.. code:: ipython3

    print(f"Gate fidelity: {gate_fid.get_value(times)}")


.. parsed-literal::

    Gate fidelity: 0.9593122908985877


References
----------

- **(Shahriari et al., 2016)** B. Shahriari et al., “Taking the human
  out of the loop: A review of Bayesian optimization,” *Proceedings of
  the IEEE* **104**, 148–175 (2016).
