#include <algorithm>
#include <array>
#include <cmath>
#include <random>

#include <spdlog/spdlog.h>

#include "Labs/1-Drawing2D/tasks.h"

using VCX::Labs::Common::ImageRGB;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> distribution(-0.5, 0.5);

namespace VCX::Labs::Drawing2D {
    /******************* 1.Image Dithering *****************/
    void DitheringThreshold(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringRandomUniform(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                float value = input.At(x, y).r;
                float a = distribution(gen);
                value = value + a > 0.5 ? 1 : 0;
                output.At(x, y) = { value, value, value };
            }
    }

    void DitheringRandomBlueNoise(
        ImageRGB &       output,
        ImageRGB const & input,
        ImageRGB const & noise) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                color += noise.At(x, y) - glm::vec3(0.5f);
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringOrdered(
        ImageRGB &       output,
        ImageRGB const & input) {
        float mat[3][3]={
            {0.6, 0.8, 0.4},
            {0.1, 0.0, 0.3},
            {0.5, 0.2, 0.7},
        };
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                for(std::size_t i = 0; i <= 2; i ++){
                    for(std::size_t j = 0; j <= 2; j ++) {
                        float value = color.r > mat[i][j] ? 1 : 0;
                        output.At(x * 3 + i, y * 3 + j) = {value, value, value};
                    }
                }
            }
    }

    void DitheringErrorDiffuse(
        ImageRGB &       output,
        ImageRGB const & input) {
        output = input;
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = output.At(x, y);
                float value = color.r;
                float value_upd = value > 0.5 ? 1 : 0;
                glm::vec3 err = {value - value_upd, value - value_upd, value - value_upd};
                if(y + 1 < input.GetSizeY()){
                    glm::vec3 tmp = output.At(x, y + 1);
                    output.At(x, y + 1) = tmp + err * (7.0f / 16.0f);
                }
                if(x + 1 < input.GetSizeX() && y > 0){
                    glm::vec3 tmp = output.At(x + 1, y - 1);
                    output.At(x + 1, y - 1) = tmp + err * (3.0f / 16.0f);
                }
                if(x + 1 < input.GetSizeX()){
                    glm::vec3 tmp = output.At(x + 1, y);
                    output.At(x + 1, y) = tmp + err * (5.0f / 16.0f);
                }
                if(x + 1 < input.GetSizeX() && y + 1 < input.GetSizeY()){
                    glm::vec3 tmp = output.At(x + 1, y + 1);
                    output.At(x + 1, y + 1) = tmp + err * (1.0f / 16.0f);
                }
                output.At(x, y) = {value_upd, value_upd, value_upd};
            }
    }

    /******************* 2.Image Filtering *****************/
    void Blur(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                if(x == 0 || x + 1 == input.GetSizeX() || y == 0 || y + 1 == input.GetSizeY()) output.At(x, y) = input.At(x, y);
                else{
                    glm::vec3 color = {0, 0, 0};
                    for(int i = -1; i <= 1; i ++) 
                        for(int j = -1; j <= 1; j ++){
                            color += input.At(x + i, y + j);
                        }
                    color /= 9;
                    output.At(x, y) = color;
                }
            }
    }

    void Edge(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        float filter[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                if(x == 0 || x + 1 == input.GetSizeX() || y == 0 || y + 1 == input.GetSizeY()) output.At(x, y) = input.At(x, y);
                else{
                    glm::vec3 color = {0, 0, 0};
                    for(int i = -1; i <= 1; i ++) 
                        for(int j = -1; j <= 1; j ++){
                            color += input.At(x + i, y + j) * filter[i + 1][j + 1];
                        }
                    output.At(x, y) = glm::abs(color);
                }
            }
    }

    /******************* 3. Image Inpainting *****************/
    void Inpainting(
        ImageRGB &         output,
        ImageRGB const &   inputBack,
        ImageRGB const &   inputFront,
        const glm::ivec2 & offset) {
        output             = inputBack;
        std::size_t width  = inputFront.GetSizeX();
        std::size_t height = inputFront.GetSizeY();
        glm::vec3 * g      = new glm::vec3[width * height];
        memset(g, 0, sizeof(glm::vec3) * width * height);
        // set boundary condition
        for (std::size_t y = 0; y < height; ++y) {
            g[y * width] = inputBack.At(offset.x, offset.y + y) - inputFront.At(0, y);
            g[y * width + width - 1] = inputBack.At(offset.x + width - 1, offset.y + y) - inputFront.At(width - 1, y);
            // set boundary for (0, y), your code: g[y * width] = ?
            // set boundary for (width - 1, y), your code: g[y * width + width - 1] = ?
        }
        for (std::size_t x = 0; x < width; ++x) {
            g[x] = inputBack.At(offset.x + x, offset.y) - inputFront.At(x, 0);
            g[(height - 1) * width + x] = inputBack.At(offset.x + x, offset.y + height - 1) - inputFront.At(x, height - 1);
            // set boundary for (x, 0), your code: g[x] = ?
            // set boundary for (x, height - 1), your code: g[(height - 1) * width + x] = ?
        }

        // Jacobi iteration, solve Ag = b
        for (int iter = 0; iter < 8000; ++iter) {
            for (std::size_t y = 1; y < height - 1; ++y)
                for (std::size_t x = 1; x < width - 1; ++x) {
                    g[y * width + x] = (g[(y - 1) * width + x] + g[(y + 1) * width + x] + g[y * width + x - 1] + g[y * width + x + 1]);
                    g[y * width + x] = g[y * width + x] * glm::vec3(0.25);
                }
        }

        for (std::size_t y = 0; y < inputFront.GetSizeY(); ++y)
            for (std::size_t x = 0; x < inputFront.GetSizeX(); ++x) {
                glm::vec3 color = g[y * width + x] + inputFront.At(x, y);
                output.At(x + offset.x, y + offset.y) = color;
            }
        delete[] g;
    }

    /******************* 4. Line Drawing *****************/
    void DrawLine(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1) {
        int x0 = p0.x;
        int y0 = p0.y;
        int x1 = p1.x;
        int y1 = p1.y;
        int dx = std::abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            canvas.At(x0, y0) = color;
            if (x0 == x1 && y0 == y1)
                break;
            int error2 = 2 * error;
            if (error2 >= dy) {
                error += dy;
                x0 += sx;
            }
            if (error2 <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    /******************* 5. Triangle Drawing *****************/
    void DrawTriangleFilled(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1,
        glm::ivec2 const p2) {
        std::array<glm::ivec2, 3> const vertices { p0, p1, p2 };
        int const minY = std::max(0, std::min({ p0.y, p1.y, p2.y }));
        int const maxY = std::min(
            static_cast<int>(canvas.GetSizeY()) - 1,
            std::max({ p0.y, p1.y, p2.y }));

        for (int y = minY; y <= maxY; ++y) {
            float minX             = 0.0f;
            float maxX             = 0.0f;
            bool  hasIntersection  = false;

            for (std::size_t edge = 0; edge < vertices.size(); ++edge) {
                glm::ivec2 const a = vertices[edge];
                glm::ivec2 const b = vertices[(edge + 1) % vertices.size()];

                if (a.y == b.y) {
                    if (y != a.y) continue;
                    float const edgeMinX = static_cast<float>(std::min(a.x, b.x));
                    float const edgeMaxX = static_cast<float>(std::max(a.x, b.x));
                    if (! hasIntersection) {
                        minX = edgeMinX;
                        maxX = edgeMaxX;
                        hasIntersection = true;
                    } else {
                        minX = std::min(minX, edgeMinX);
                        maxX = std::max(maxX, edgeMaxX);
                    }
                    continue;
                }

                int const edgeMinY = std::min(a.y, b.y);
                int const edgeMaxY = std::max(a.y, b.y);
                if (y < edgeMinY || y > edgeMaxY) continue;

                float const t = static_cast<float>(y - a.y) / static_cast<float>(b.y - a.y);
                float const x = static_cast<float>(a.x) + t * static_cast<float>(b.x - a.x);
                if (! hasIntersection) {
                    minX = x;
                    maxX = x;
                    hasIntersection = true;
                } else {
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                }
            }

            if (! hasIntersection) continue;

            int const startX = std::max(0, static_cast<int>(std::ceil(minX)));
            int const endX   = std::min(
                static_cast<int>(canvas.GetSizeX()) - 1,
                static_cast<int>(std::floor(maxX)));
            for (int x = startX; x <= endX; ++x)
                canvas.At(x, y) = color;
        }
    }

    /******************* 6. Image Supersampling *****************/
    void Supersample(
        ImageRGB &       output,
        ImageRGB const & input,
        int              rate) {
        int Hin = input.GetSizeX();
        int Win = input.GetSizeY();
        int Hout = output.GetSizeX();
        int Wout = output.GetSizeY();
        for (std::size_t x = 0; x < Hout; ++x)
            for (std::size_t y = 0; y < Wout; ++y) {
                glm::vec3 color = {0, 0, 0};
                for(int i = 0; i < rate; i ++){
                    for(int j = 0; j < rate; j ++){
                        float source_x = (x + (i + 0.5f) / rate) * Win / Wout - 0.5f;
                        float source_y = (y + (j + 0.5f) / rate) * Win / Wout - 0.5f;
                        if(source_x < 0)source_x = 0;
                        if(source_x > Hin - 1)source_x = Hin - 1;
                        if(source_y < 0)source_y = 0;
                        if(source_y > Win - 1)source_y = Win - 1;
                        int x0 = std::min((int)source_x, Hin - 2);
                        int x1 = x0 + 1;
                        int y0 = std::min((int)source_y, Win - 2);
                        int y1 = y0 + 1;
                        color += (source_x - x0) * (source_y - y0) * input.At(x1, y1);
                        color += (source_x - x0) * (y1 - source_y) * input.At(x1, y0);
                        color += (x1 - source_x) * (source_y - y0) * input.At(x0, y1);
                        color += (x1 - source_x) * (y1 - source_y) * input.At(x0, y0);
                    }
                }
                color /= (rate * rate);
                output.At(x, y) = color;
            }
        
    }

    /******************* 7. Bezier Curve *****************/
    // Note: Please finish the function [DrawLine] before trying this part.
    glm::vec2 CalculateBezierPoint(
        std::span<glm::vec2> points,
        float const          t) {
        size_t cnt = points.size();
        std::vector<glm::vec2> tmp(points.begin(), points.end());
        for(int i = 1; i < cnt ; i ++){
            for(int j = 0; j < cnt - i; j ++){
                tmp[j] = (1 - t) * tmp[j] + t * tmp[j + 1];
            }
        }
        return tmp[0];
    }
} // namespace VCX::Labs::Drawing2D
