/** @type {import('next').NextConfig} */
const nextConfig = {
  
  experimental: {
      // 禁用可能导致兼容性问题的实验性功能
    esmExternals: false,
    // 启用传统浏览器支持
    legacyBrowsers: true,
    // 启用最新的实验性功能
    turbo: {
      rules: {
        '*.svg': {
          loaders: ['@svgr/webpack'],
          as: '*.js',
        },
      },
    },
  },
  // Webpack 配置用于兼容性处理
  // webpack: (config, { dev, isServer }) => {
  //   // 添加 polyfills
  //   if (!isServer) {
  //     config.resolve.fallback = {
  //       ...config.resolve.fallback,
  //       fs: false,
  //       net: false,
  //       tls: false,
  //       crypto: false,
  //       stream: false,
  //       url: false,
  //       zlib: false,
  //       http: false,
  //       https: false,
  //       assert: false,
  //       os: false,
  //       path: false,
  //     };
      
  //     // 添加全局变量 polyfill
  //     config.plugins.push(
  //       new config.webpack.DefinePlugin({
  //         'globalThis': 'window',
  //         'global': 'window',
  //       })
  //     );
  //   }
    
  //   // SVG 处理
  //   config.module.rules.push({
  //     test: /\.svg$/,
  //     use: ['@svgr/webpack'],
  //   });
    
  //   return config;
  // },
  
  // Babel 配置用于语法降级
  babel: {
    presets: [
      [
        'next/babel',
        {
          'preset-env': {
            targets: {
              browsers: [
                '>1%',
                'last 4 versions',
                'Firefox ESR',
                'not ie < 9',
                'not dead'
              ]
            },
            useBuiltIns: 'usage',
            corejs: 3
          }
        }
      ]
    ]
  },
  // 优化配置
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },
  // 图像优化
  images: {
    formats: ['image/webp', 'image/avif'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    unoptimized: true,
  },
  // 性能优化
  // swcMinify: true,
  // 输出配置 - 修改为静态导出
  //basePath: './',
  assetPrefix:'./',
  trailingSlash: true,
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // 静态导出配置
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
}

export default nextConfig
