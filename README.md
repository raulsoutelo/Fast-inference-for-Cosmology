# Fast-inference-for-Cosmology

In this project I have implemented five different methods to explore the parameters of a cosmological model using MCMC methods (all of
them are variants of the Metropolis-Hastings method). Cosmological models have two important characteristics:
1) Evaluating the function depends on the parameters that are modified with respect to the previous evaluation (fast and slow parameters)
2) Some parameters are not of interest (nuisance parameters)

These five methods are intended to exploit these two features, so that the estimation of the probability distribution of the parameters of interest is as efficient as possible.

The methods are:
1) Metropolis-Hastings: 
@article{hastings1970monte,
  title={Monte Carlo sampling methods using Markov chains and their applications},
  author={Hastings, W Keith},
  journal={Biometrika},
  volume={57},
  number={1},
  pages={97--109},
  year={1970},
  publisher={Biometrika Trust}
}
2) Extra update Metropolis:
@article{lewis2002cosmological,
  title={Cosmological parameters from CMB and other data: A Monte Carlo approach},
  author={Lewis, Antony and Bridle, Sarah},
  journal={Physical Review D},
  volume={66},
  number={10},
  pages={103511},
  year={2002},
  publisher={APS}
}
3) Fast-slow decorrelation:
@article{lewis2013efficient,
  title={Efficient sampling of fast and slow cosmological parameters},
  author={Lewis, Antony},
  journal={Physical Review D},
  volume={87},
  number={10},
  pages={103529},
  year={2013},
  publisher={APS}
}
4) APM1 MI+MH and 5) APM2 MI+MH. Two variants of the Pseudo-Marginal approach proposed by Iain Murray and Matt Graham:
@inproceedings{murray2016pseudo,
  title={Pseudo-marginal slice sampling}, 
  author={Murray, Iain and Graham, Matthew},
  booktitle={Artificial Intelligence and Statistics},
  pages={911--919},
  year={2016}
}

Two probability distributions have been explored with these methods:
1) A simple test distribution (implemented in Toy_Posterior_Class)
2) And a more complicated distribution, intended to be representative of a cosmological model realizable by the CosmoSIS package (Real_Posterior_Class)

A more detailed explanation of the methods implemented and their performance on these two probability distributions is found in 
dissertation_RaulSoutelo.pdf.

In order to run the code, it is enough to run the script main.py (installing the emcee package is needed). Creating the object P from the class Toy_Posterior or from the class Real_Posterior, we can explore the two different probability distributions. In order to use the code to explore a different probability distribution, we need to create a new class with the same attributes and methods than these ones.
