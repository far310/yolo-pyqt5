/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
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
  //trailingSlash: true,
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
