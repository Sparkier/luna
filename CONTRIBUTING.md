## Contributing

Recommendations and requirements for how to best contribute to **Luna**.
We greatly appreciate any contribution and we hope these guidelines make it easier for you to get started.

### Committing

You cannot directly commit to the master branch of this repository.
To contribute your changes, you need to apply them to a different branch (on a for or on this repo), and then create a pull request.
We will review these requests as fast as we can to get your changes in.
Please make sure your changes pass our tests (they will be run automatically for every PR).

### Open Issues

For contribution opportunities, see our [Issues](https://github.com/Sparkier/luna/issues).
To get started, look for ones with the tag **good first issue**.

### Releasing

To create a new release of Luna, update the verion number in [setup.cfg](./setup.cfg) and tag your release accordingly with `git tag <version_number>`.
Make sure to push your tag with `git push --tags`.
Once your commit is merged, it will automatically be deployed.

### Docs

To also deploy the documentation if something has changed there, use the [update script](./pages-publish.sh).
