import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
        destination:
          process.env.NODE_ENV === 'development'
            ? 'http://127.0.0.1:5328/api/:path*'
            : '/api/',
      },
    ]
  },
  turbopack: {
    rules: {
      '**/*.svg': {
        "loaders": ["@svgr/webpack"],
        "as": "*.js"
      },
      '**/*.html': {
        "loaders": ["html-loader"],
        "as": "*.js"
      }
    }
  },
  webpack: (config) => {
    config.module.rules.push({
      test: /\.svg$/,
      use: ["@svgr/webpack"],
    });
    config.module.rules.push({
      test: /\.html$/,
      use: "html-loader",
    });
    return config;
  },
  experimental: {
    reactCompiler: true,
  }
};

export default nextConfig;