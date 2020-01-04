# Collaboration

This project uses a novel methodology for development, in which you only need [Docker installed](https://docs.docker.com/install/).
Fork the project, clone, and you'll find a `docker` folder and a `docker-compose.yml` file in the project root.
We provided [packaged testing environments](https://hub.docker.com/orgs/auditorium) (in the form of Docker images) for all the Python versions we target.
There is also a `makefile` with all the necessary commands.

The workflow is something like this:

* Fork, clone, and make some changes.
* Run `make` to run the local, fast tests. The first time this will download the corresponding image.
* Fix errors (if any) and watch the testing coverage. Make sure to at least cover the newly added features.
* Run `make test-full` to run the local but long tests. This will download all the remaining images for each Python environment.
* If all worked, push and pull-request.

If you need to tinker with the dev environment, `make shell` will open a shell inside the latest Python environment where you can run and test commands.

This project uses [poetry](https://python-poetry.org/) for package management. If you need to install new dependencies, run `make shell` and then `poetry add ...` inside the dockerized environment. Finally, don't forget to `poetry lock` and commit the changes to `pyproject.toml` and `poetry.lock` files.

## License

License is MIT, so you know the drill: fork, develop, add tests, pull request, rinse and repeat.

> MIT License
>
> Copyright (c) 2019-2020 Suilan Estevez-Velarde and contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
