"""Each tab builds itself and wires its own events.

One module per tab rather than one big `build_app`: a tab is the natural unit of
change here, and keeping the event wiring next to the components it wires is what
stops a 600-line Blocks function.
"""
