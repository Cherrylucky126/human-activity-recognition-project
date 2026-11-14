"""HAR wrist unit — live dashboard.

PR 1 of 14: application skeleton only.

Serves a titled, empty page so that every later PR has somewhere to land. No
data source, no callbacks, no panels yet.

Run it with::

    cd dashboard
    python app.py

then open http://127.0.0.1:8050 in a browser.
"""

from dash import Dash, html

from config import APP_TITLE, HOST, PORT

app = Dash(__name__, title=APP_TITLE)

app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
        "maxWidth": "960px",
        "margin": "0 auto",
        "padding": "32px 16px",
    },
    children=[
        html.H1(APP_TITLE, style={"marginBottom": "4px"}),
        html.P(
            "Skeleton only. Panels arrive in PR 4, live data in PR 2 and PR 3.",
            style={"color": "#666", "marginTop": 0},
        ),
    ],
)


if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
