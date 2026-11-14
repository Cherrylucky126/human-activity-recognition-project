"""HAR wrist unit — live dashboard.

Wiring only: this module builds the Dash app, attaches the layout, and from
PR 3 onward will hold the callbacks that push live values into it. The shape of
the page lives in ``layout.py``.

Run it with::

    cd dashboard
    python app.py

then open http://127.0.0.1:8050 in a browser.

Progress: PR 4 of 14. The page shows every region it will ever have, all empty.
No data source is attached yet.
"""

from dash import Dash

from config import APP_TITLE, HOST, PORT
from layout import build_layout

app = Dash(__name__, title=APP_TITLE)
app.layout = build_layout()

# Dash warns about callbacks that target ids which do not exist yet. Every id
# in layout.Ids is present from PR 4 on, so nothing is suppressed here; if a
# later PR needs an id that is created dynamically, set
# suppress_callback_exceptions=True above rather than inventing hidden divs.


if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
