window.MathJax = {
    tex: {
        tags: "ams",
        macros: {
            x: "{\\times}",
            bm: ["{\\boldsymbol{#1}}", 1],
            dd: ["{\\frac{\\partial #1}{\\partial #2}}", 2]
        }
    },
    chtml: {
        scale: 0.9,
        mtextInheritFont: true
    }
}; 