## Acknowledgements and Code Origins

Parts of the nodal-to-modal transformation and SIAC postprocessing implementation are adapted from the MATLAB SIAC test codes by Jennifer K. Ryan:

https://github.com/jennkryan/SIACMagicTestCodes/

In particular, the construction of Vandermonde-based transformations and SIAC kernel evaluation follow the structure and ideas of these reference implementations, but have been reimplemented and extende here for:

- tensor-product DG representations,
- 2D image/tomographic data,
- Python-based workflows.