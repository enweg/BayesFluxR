# If we are in a GitHub Action, we need to install Julia
test_setup <- function(...) {
  if (Sys.getenv("CI") != ""){
    BayesFluxR_setup(installJulia = TRUE, env_path = ".", ...)
  } else {
    choice <- Sys.getenv("BayesFluxTestInstallJulia")
    if (choice == ""){
      choice <- readline(prompt = "Install Julia for tests? Y/N: ")
      Sys.setenv("BayesFluxTestInstallJulia" = choice)
    }
    if (choice %in% c("Y", "y", "yes", "Yes")) {
      BayesFluxR_setup(installJulia = TRUE, env_path = ".", ...)
    }
    else {
      BayesFluxR_setup(installJulia = FALSE, env_path = ".", ...)
    }
  }
}
