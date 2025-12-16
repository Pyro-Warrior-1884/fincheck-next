"use client"

export default function ChartWrapper({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-2xl border bg-white p-6 shadow-sm space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-gray-800">
          {title}
        </h3>
        {description && (
          <p className="text-xs text-gray-500">
            {description}
          </p>
        )}
      </div>
      {children}
    </div>
  )
}
