import {build} from 'esbuild';
await build({entryPoints:['web/app.js'],bundle:true,minify:true,outfile:'web/bundle.js',format:'esm'});
