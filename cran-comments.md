## R CMD check results

!! This is a new submission

There were no ERRORs and no WARNINGs

There was 0 Notes:

## Previous CRAN maintainer comments: 

> Thanks, but please single quote software names in the Description field, 
e.g. 'BayesFlux.jl'.
> Please fix and resubmit.
> Is there some reference about the method you can add in the Description 
field in the form Authors (year) <doi:10.....> or <arXiv:.....>?
> Best,
> Uwe Ligges

Reply: All software names have now been single quoted in the Description field. References to the relevant literature exist in the README and in the documentation to the Julia functions. 


> Thanks, we see:
> Found the following (possibly) invalid URLs:
> URL: http://arxiv.org/abs/2102.01691 (moved to 
> https://arxiv.org/abs/2102.01691)
> From: README.md
> Status: 301
> Message: Moved Permanently
>
> Please change http --> https, add trailing slashes, or follow moved 
content as appropriate.

Reply: The link has now been fixed. 

> We see
> ./R/BayesFluxR.R:  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)
> which is apparently executed dfuring the checks. Please folow the CRAN 
> policies. It is not permitted to install thirs party software without 
> explicit user interaction! Particularly not in examples or test!

Reply: Following other packages such as 'diffeqr', tests involving Julia are skipped on CRAN now. 
