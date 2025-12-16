"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"
import { chartTheme } from "./chartTheme"

type ChartItem = {
  model: string
  confidence: number
}

type Props = {
  data: ChartItem[]
}

export default function ChartSection({ data }: Props) {
  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm transition hover:shadow-md">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">
        Model Confidence Comparison
      </h2>

      <ResponsiveContainer width="100%" height={340}>
        <BarChart
          data={data}
          margin={{ top: 20, right: 20, left: 0, bottom: 10 }}
        >
          <defs>
            <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="0%"
                stopColor={chartTheme.colors.primaryLight}
              />
              <stop
                offset="100%"
                stopColor={chartTheme.colors.primary}
              />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke={chartTheme.colors.grid}
          />

          <XAxis
            dataKey="model"
            tick={{ fill: chartTheme.colors.text, fontSize: 12 }}
          />

          <YAxis
            domain={[0, 1]}
            tick={{ fill: chartTheme.colors.text, fontSize: 12 }}
            tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
          />

          <Tooltip
            formatter={(value) => {
              if (typeof value === "number") {
                return `${Math.round(value * 100)}%`
              }
              return ""
            }}
            contentStyle={{
              backgroundColor: chartTheme.colors.tooltipBg,
              borderRadius: "12px",
              border: "none",
              color: chartTheme.colors.tooltipText,
              fontSize: "12px",
            }}
            labelStyle={{ color: "#9ca3af" }}
            cursor={{ fill: "rgba(99,102,241,0.08)" }}
          />

          <Bar
            dataKey="confidence"
            fill="url(#barGradient)"
            radius={[10, 10, 0, 0]}
            animationDuration={800}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
