# Copyright 2024 SAFRAN SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ----------------
# Exercice #1
# ----------------

from scipy.signal import find_peaks
peaks, _ = find_peaks(amplitudes[13])

for index, peak in enumerate(peaks):
    y_harm = sols[13][indexH, peak]
    harm_contrib = np.zeros(((len(y_harm) - 1) // 2,)) # retrieves nh
    for i in range(len(harm_contrib)):
        harm_contrib[i] = np.linalg.norm(y_harm[2*i+1:2*i+3])

    # List of x-tick labels
    xticks_labels = np.arange(1,14,1)

    # Plotting the bar graph
    plt.bar(range(len(harm_contrib)), harm_contrib, tick_label=xticks_labels)

    # Set x-tick label fontsize
    plt.xticks(fontsize=12)  # Change the fontsize as needed

    plt.xlabel('Harmonics of the fundamental frequency')
    plt.ylabel('Amplitude [m]')
    plt.title('Harmonic content of the response at peak #'+str(index+1))

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')

    # Show the plot
    plt.show()
	
# ----------------
# Exercice #2
# ----------------

y0 = [1.0,0.0]
nit = 2048

om = np.linspace(INP['analysis']['FRF']['puls_inf'], INP['analysis']['FRF']['puls_sup'], len(omegas[13]))
n_period_all = []
t_all = []
y_all = []

for omega in om:
    test = False
    n_period = 0
    while not(test):
        n_period += 5 # a smaller step can be chosen
        t, y = time_integration(duffing, omega, y0, nit, n_period)
        # Test to check steady-state
        test = np.abs(np.linalg.norm(y[-nit:])-np.linalg.norm(y[-2*nit:-nit]))/np.linalg.norm(y[-nit:])<1e-3
    n_period_all.append(n_period)
    t_all.append(t)
    y_all.append(y)