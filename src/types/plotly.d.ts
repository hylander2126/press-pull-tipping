declare module "plotly.js-basic-dist-min" {
  const Plotly: {
    react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
  };
  export default Plotly;
}
