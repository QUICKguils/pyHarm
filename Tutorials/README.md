# Module Tutorials

This pyHarm module consists of a series of Jupyter Notebooks that are built in the objective of understanding Harmonic Balance Method and its key characteristics. The tutorials are built in thematic blocks. The blocks are described in the sections below.

## Block 01 : Introduction to Harmonic Balance Method
This block regroups three tutorials that introduce basic concepts for the HBM and treating them using pyHarm in an inductive way. A supplementary tutorial is given as well in order to challenge the learner. The Table below details the theme introduced by each tutorial. 

| Tuto | Name | Summary |
| :- | :- | :- | 
| 001 | Time Intergration | This tutorial introduces the HBM method and tries to weight the pros and cons of the method versus the classical time integration method in the linear case. | 
| 002 | Alternating Frequency Time scheme | This tutorial introduces the AFT scheme used for computing non-linear functions that are explicit in the time domain. It uses the simple pendulum as an exemple that drives all the tutorial. |
| 003 | Predictor/Corrector | This tutorial introduces the prediction/correction scheme used in the `FRF_NonLinear` analysis of pyHarm. This shows the impact of using a continuation method for convergence. Some user-made predictors are coded and compared to more complex and efficient state of the art predictor methods. |
| 004 | SkyScrapper | This tutorial is a challenging tutorial to conclude the first block of tutorial. It consists of a whole study over a skyscrapper represented by a lumped model. In this tutorial, custom user forcing are coded in order to represent earthquake and wind loadings and damping systems are added to the building to damp the main resonances. |


## Block 02 : Introduction to friction phenomenon
This block regroups three tutorials that introduce the friction phenomenon in an HBM framework. It is advised to do this block after the previous one as many components are important. The Table below details the theme introduced by each tutorial. 

| Tuto | Name | Summary |
| :- | :- | :- | 
| 005 | Introduction to friction | This tutorial introduces the friction effects through the use of a $`\tanh{}`$ function. This also introduces the coding of a non-linear element into pyHarm for the first time. The effect of friction over the damping is studied on a simple test case in order to see the sensitivity of the damping effect with the constitutive parameters of friction. |
| 006 | Bladed disk with ring damper | This tutorial treats a lumped mass model of a full bladed disk. A ring damper is added to the bladed disk in order to dampen resonances. |
| 007 | Super-element and 3D blade | This tutorial shows the Craig-Bampton methodology for reducing a dynamical model. The method is applied onto a simple beam test case and the performance of the reduced order model is compared with the full order model. A forced response analysis is then made on a provided super-element of a 3D academic blade |



## Block -  non-regression test base
This block is constituted of a single Jupyter Notebook file and constains all the test cases that are present in the non-regression test base. It can be used to see the look of the Forced Response Curves of those test cases as well as seing examples of inputs for pyHarm. 
