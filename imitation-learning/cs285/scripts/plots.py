import matplotlib.pyplot as plt
import numpy as np

"""
Figure 1: Part 3.2

Experiment with one set of hyperparameters that affects the performance of the behavioral cloning agent, such as the amount of
training steps, the amount of expert data provided, or something that you come up with yourself. For one of the tasks used in
the previous question, show a graph of how the BC agent’s performance varies with the value of this hyperparameter. In the
caption for the graph, state the hyperparameter and a brief rationale for why you chose it.

Task: Ant
Hyperparameter chosen: --num_agent_train_steps_per_iter
Rationale: as the number of agent training steps per iteration increases, the model will acquire a greater repository of
training data and the chance of seeing unseen states increases. Thus, this increases the amount of training data that the
model can train on and the reach of the space of the training set via greater opportunity for exploration.


params = [250, 500, 750, 1000, 1250, 1500]
avg_returns = [562.1510009765625, 3032.5703125, 3982.265380859375, 4647.76220703125, 4553.966796875, 4599.3115234375]
inits = [4681.891673935816, 4681.891673935816, 4681.891673935816, 4681.891673935816, 4681.891673935816, 4681.891673935816]
performance = [avg_returns[i]/inits[i] for i in range(len(inits))]
print("plotting")
plt.title("Performance over Training Steps per Iteration")
plt.plot(params,performance)
plt.xlabel('Training Steps per Iteration')
plt.ylabel('Performance')
plt.show()
plt.close()

"""

"""
Run DAgger and report results on the two tasks you tested previously with behavioral cloning. Report
your results in the form of a learning curve, plotting the number of DAgger iterations vs. the policy’s
mean return, with error bars to show the standard deviation. Include the performance of the expert
policy and the behavioral cloning agent on the same plot (as horizontal lines that go across the plot). In
the caption, state which task you used, and any details regarding network architecture, amount of data,
etc. (as in the previous section).
"""

# for first task and then same pipeline for second task

ant = {}
ant["num_iters"] = [5, 7, 10, 15, 20]
ant["mean_ret"] = [4384.861328125, 4715.3408203125, 4626.396484375, 4581.64453125, 4707.69189453125]
ant_bc = [4647.76220703125]*len(ant["num_iters"])
ant_expert = [4681.891673935816]*len(ant["num_iters"])

hop = {}
hop["num_iters"] = [5, 7, 10, 15, 20]
hop["mean_ret"] = [3702.5498046875, 2167.072021484375, 3709.63427734375, 3728.9716796875, 3721.44775390625]
hop_bc = [856.6791381835938]*len(ant["num_iters"])
hop_expert = [3717.5129936182307]*len(ant["num_iters"])

# ant, 10: 4626.396484375
plt.title("Number of DAgger Iterations vs. Mean Return")
plt.errorbar(ant["num_iters"], ant["mean_ret"], yerr=np.std(ant["mean_ret"]), label="Ant")
plt.errorbar(ant["num_iters"], ant_bc, yerr=140.6004180908203, label="Ant BC")
plt.errorbar(hop["num_iters"], hop["mean_ret"], yerr=np.std(hop["mean_ret"]), label="Hopper")
plt.plot(ant["num_iters"], ant_expert, label="Ant Expert")
plt.errorbar(hop["num_iters"], hop_bc, yerr=204.69802856445312, label="Hopper BC")
plt.plot(ant["num_iters"], hop_expert, label="Hopper Expert")
plt.legend()
# need error bars for standard deviation
plt.xlabel('Number of DAgger Iterations')
plt.ylabel('Mean Return')
plt.ylim(0,5000)
# performance of the expert policy and the behavioral cloning agent on the same plot (as horizontal lines that go across the plot)
plt.show()
plt.close()