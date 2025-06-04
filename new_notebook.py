import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    # Welcome to the Enverge
    return


app._unparsable_cell(
    r"""
    Example of retrieving Marimo API
    """,
    name="_"
)


@app.cell
def _():
    import requests
    # NB: The service is running a the `/notebook` base URL, so all requests must use this prefix
    resp = requests.get("http://localhost:8000/notebook/api/usage")
    resp.json()
    return


if __name__ == "__main__":
    app.run()
