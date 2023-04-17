DataEmbedder library
====================

The *dataembedder library* embeds data (points and sub-models) in a scaffold by defining permanent *material coordinates* over it, calculated from its geometric coordinates relative to the scaffold's fitted geometry field. Its client API is mainly concerned with selecting which annotated groups in the input data are to be embedded, then requesting the output to be generated. Model representation and calculations are performed with the underlying *Zinc library*.

Most users will use this from the ABI Mapping Tools' **Data Embedder** user interface for this library. Its documentation also applies to this back-end library.

Examples of direct usage are in the tests folder of the library's `github repository <https://github.com/ABI-Software/dataembedder>`_.
