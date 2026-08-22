"""C4: everything that speaks to a llama.cpp server.

The dispatcher and the root client are the only modules in the package
allowed to hold an HTTP client; `roottext` is their pure text shaping,
`launchlog` reads a server's own stderr, `envelope` and `leakcheck`
inspect what a leaf answered. Modules are addressed directly
(`rlm.serve.dispatcher`); nothing is re-exported, so the dependency-rule
lint keeps naming the module that actually imports a client."""
